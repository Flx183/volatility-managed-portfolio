"""Phase 7: weekly holding periods and forecasts, the screened books, the segment runner and the
tests, on simulated data only."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

from src import engine as e
from src.config import FACTORS
from src.engine import data, evaluation as ev, execution as ex, synthetic as syn
from src.engine.data import MONTH, WEEK
from src.models import filters, inputs, weekly
from src.walkforward import runs, screened, stats, tracks

YEARS = 14


def _p(years=YEARS, seed=0):
    return syn.garch(years=years, seed=seed, start="1990-01-01")["x"].rename("p")


@pytest.fixture(scope="module")
def p():
    return _p()


def _panel(years=24, seed=0):
    """Seven fake factors, a risk-free rate and the event and VIX columns, on business days."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1980-01-01", periods=252 * years, name="date")
    panel = pd.DataFrame(rng.normal(0.0003, 0.01, (len(idx), len(FACTORS))), index=idx, columns=FACTORS)
    panel["rf"] = 0.0001
    return panel


def _survivors(panel, by_year):
    """A survivor list for every year-end, from {year: [factors]} (missing years: nothing passes)."""
    from src.screen import year_ends
    ends = year_ends(panel.index)
    return pd.Series({d: by_year.get(d.year, []) for d in ends}, name="assets")


# ---------------------------------------------------------------- engine: weekly periods

def test_a_weekly_book_trades_weeks_and_its_table_measures_weeks(p):
    rf = pd.Series(0.0001, p.index)
    book = e.Book.single(p, rf, "A", freq=WEEK)
    assert book.freq == WEEK and book.positions.index.freqstr == "W-FRI"
    tab = data.table(p, rf, WEEK)
    wk = p.index.to_period(WEEK)
    first = wk[0]
    x, r = p[wk == first], rf[wk == first]
    assert tab.loc[first, "ret"] == pytest.approx(np.prod(1 + x + r) - np.prod(1 + r), rel=1e-12)
    assert tab.loc[first, "rv"] == pytest.approx((x ** 2).sum(), rel=1e-12)
    assert (ex.period_labels(p.index, 0, WEEK) == wk).all()


def test_holding_one_unit_a_week_earns_the_asset(p):
    rf = pd.Series(0.0001, p.index)
    per = ex.holding_periods(e.Book.single(p, rf, "A", freq=WEEK))
    weeks = per.growth.index[1:30]
    out = ex.simulate(per, pd.DataFrame({"A": 1.0}, index=weeks))
    assert np.allclose(out["net"], per.growth.loc[weeks, "A"] - 1)
    with pytest.raises(ValueError, match="consecutive"):
        ex.simulate(per, pd.DataFrame({"A": 1.0}, index=weeks.delete(4)))


def test_an_instrument_held_at_zero_needs_no_return():
    """The check for missing returns looks only at held instruments (a Phase 7 fix: it used to flag
    every instrument held at zero)."""
    x = _p(3)
    y = x.copy()
    y.loc["1991-03"] = np.nan
    rf = pd.Series(0.0001, x.index)
    months = pd.period_range("1991-02", periods=3, freq="M")
    book = e.Book(pd.DataFrame({"A": x, "B": y}), rf, pd.DataFrame({"A": 1.0, "B": 0.0}, index=months))
    per = ex.holding_periods(book)
    out = ex.simulate(per, book.positions)
    assert np.allclose(out["net"], per.growth.loc[months, "A"] - 1)
    with pytest.raises(ValueError, match="no return in 1991-03"):
        ex.simulate(per, book.positions.assign(B=0.5))


# ---------------------------------------------------------------- weekly forecasts

def test_weekly_origins_start_with_the_monthly_ones_and_use_the_latest_fit(p):
    wo, fit_at = weekly.schedule(p)
    mo, _ = inputs.schedule(p)
    assert wo.iloc[0] >= mo.iloc[0] and (wo.iloc[0] == mo.iloc[0] or weekly.origins(p).shift(1).loc[wo.index[0]] < mo.iloc[0])
    assert (fit_at <= wo.to_numpy()).all()
    nxt = np.searchsorted(mo.to_numpy(), fit_at, side="right")           # the refit after the one used
    assert all(n == len(mo) or mo.iloc[n] > d for n, d in zip(nxt, wo.to_numpy()))
    assert (p.index[wo.to_numpy()] == pd.Series(p.index, p.index).groupby(p.index.to_period(WEEK)).max().loc[wo.index]).all()
    _, fit_year = weekly.schedule(p, "year")
    assert set(p.index[np.unique(fit_year)].month[1:]) == {12}


def test_weekly_rolling_and_ewma_measure_what_they_say(p):
    wo, _ = weekly.schedule(p)
    d = wo.iloc[5]
    assert weekly.rolling(p).iloc[5] == pytest.approx((p.iloc[d - 4:d + 1] ** 2).sum(), rel=1e-12)
    s2 = (p.iloc[0] ** 2)
    for r in p.iloc[: d + 1].to_numpy():
        s2 = 0.94 * s2 + 0.06 * r ** 2
    assert weekly.ewma(p).iloc[5] == pytest.approx(5 * s2, rel=1e-6)


@pytest.mark.parametrize("gjr", [False, True])
def test_weekly_garch_at_22_days_is_the_monthly_forecast_on_month_end_weeks(p, gjr):
    f, params = filters.garch_fits(p, gjr)
    w22 = weekly.garch(p, gjr, params, h=22)
    mo, _ = inputs.schedule(p)
    wo, _ = weekly.schedule(p)
    same = [(w, m) for w, d in wo.items() for m, R in mo.items() if d == R]
    assert len(same) >= 5
    assert all(w22[w] == pytest.approx(f[m], rel=1e-12) for w, m in same)
    w5 = weekly.garch(p, gjr, params)
    assert (w5 < w22).all() and (w5 > 0).all()


def test_weekly_har_is_the_bias_corrected_fit_on_rows_whose_5_day_target_is_known(p):
    f, fits = weekly.har_fits(p)
    wo, fit_at = weekly.schedule(p)
    i = 20
    d, R = wo.iloc[i], fit_at[i]
    X = inputs.har_design(p)
    y = np.log(inputs.forward(p ** 2, 5))
    rows = np.arange(R - 5 + 1)
    ok = X.iloc[rows].notna().all(axis=1).to_numpy() & np.isfinite(y.iloc[rows]).to_numpy()
    Z = np.column_stack([np.ones(ok.sum()), X.iloc[rows][ok]])
    beta = np.linalg.lstsq(Z, y.iloc[rows][ok], rcond=None)[0]
    s2 = ((y.iloc[rows][ok] - Z @ beta) ** 2).sum() / (ok.sum() - 4)
    assert f.iloc[i] == pytest.approx(np.exp(beta[0] + X.iloc[d].to_numpy() @ beta[1:] + s2 / 2), rel=1e-10)


def test_weekly_event_counts_look_5_days_ahead():
    idx = pd.bdate_range("2000-01-03", periods=60)
    panel = pd.DataFrame({"vix": 20.0, "fomc": False, "cpi": False, "jobs": False}, index=idx)
    panel.iloc[10, 1] = True                                          # an FOMC day at position 10
    x = weekly.exog(panel, idx)
    assert x["fomc"].iloc[4] == 0 and (x["fomc"].iloc[5:10] == 1).all() and x["fomc"].iloc[10] == 0


@pytest.mark.parametrize("name", ["M0", "M1", "M2", "M4", "M7"])
def test_weekly_forecasts_use_only_data_up_to_their_origin(p, name):
    rng = np.random.default_rng(1)
    cut = p.index[int(len(p) * 0.85)]
    q = p.copy()
    q[q.index > cut] = rng.normal(0, 0.05, (q.index > cut).sum())
    x_for = lambda s: weekly.exog(pd.DataFrame({"vix": 20.0, "fomc": False, "cpi": False, "jobs": False}, index=s.index), s.index)
    f = {"M0": weekly.rolling, "M1": weekly.ewma,
         "M2": lambda s: weekly.garch(s, False, filters.garch_fits(s, False)[1]),
         "M4": lambda s: weekly.har_fits(s)[0],
         "M7": lambda s: weekly.boost_fits(s, False, x_for(s))[0]}[name]
    a, b = f(p), f(q)
    known = a.index[p.index[weekly.origins(p).loc[a.index].to_numpy()] <= cut]
    assert len(known) > 20 and np.array_equal(a[known].to_numpy(), b[known].to_numpy())


# ---------------------------------------------------------------- the screened books

def test_recipes_use_only_the_listed_assets_and_skip_empty_years():
    panel = _panel()
    surv = _survivors(panel, {1992: ["mkt"], 1993: ["mkt", "bab"], 1995: ["mkt", "mom", "bab"]})
    rec = screened.recipes(panel, surv)
    assert sorted({d.year for d in rec.index.get_level_values("asof")}) == [1992, 1993, 1995]
    w = rec.filter(regex="^w_")
    assert np.allclose(w.sum(axis=1), 1)
    row = rec.loc[(pd.Timestamp("1993-12-31"), "inv_vol")]
    assert row["assets"] == "mkt bab" and row["w_mom"] == 0 and row["w_smb"] == 0
    win = panel.loc["1984-01-01":"1993-12-31", ["mkt", "bab"]]
    assert np.allclose([row["w_mkt"], row["w_bab"]], (1 / win.std()) / (1 / win.std()).sum())


def test_books_hold_each_recipe_from_the_first_rebalance_after_its_year_end():
    panel = _panel()
    surv = _survivors(panel, {1992: ["mkt"], 1993: ["mkt", "bab"], 1995: ["mkt", "mom"]})
    rec = screened.recipes(panel, surv)
    mon = screened.book(panel, rec, "inv_vol", MONTH)
    years = sorted(set(mon.positions.index.year))
    assert years == [1993, 1994, 1996]                               # 1995 follows an empty year-end
    assert (mon.positions.loc["1993", ["bab", "mom"]] == 0).all().all()
    assert np.allclose(mon.positions.loc["1994-06", ["mkt", "bab"]], rec.loc[(pd.Timestamp("1993-12-31"), "inv_vol"), ["w_mkt", "w_bab"]])
    wk = screened.book(panel, rec, "inv_vol", WEEK)
    opens = screened.opening_trades(panel.index, WEEK).loc[wk.positions.index]
    first94 = opens[opens >= pd.Timestamp("1993-12-31")].index[0]
    last93 = opens[opens < pd.Timestamp("1993-12-31")].index[-1]
    assert wk.positions.loc[first94, "bab"] > 0 and wk.positions.loc[last93, "bab"] == 0
    assert len(runs.segments(mon)) == 2 and len(runs.segments(wk)) == 2


def test_recipes_and_books_up_to_a_date_ignore_everything_after_it():
    panel = _panel()
    surv = _survivors(panel, {1992: ["mkt"], 1993: ["mkt", "bab"], 1994: ["mkt", "mom", "bab"]})
    cut = pd.Timestamp("1994-12-30")
    noisy = panel.copy()
    noisy.loc[noisy.index > cut, FACTORS] = np.random.default_rng(3).normal(0, 0.1, ((noisy.index > cut).sum(), len(FACTORS)))
    a, b = screened.recipes(panel, surv), screened.recipes(noisy, surv)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- the segment runner

def _gapped_book(freq=MONTH):
    panel = _panel(30)
    surv = _survivors(panel, {y: ["mkt", "bab"] for y in list(range(1990, 1994)) + list(range(1996, 2009))})
    rec = screened.recipes(panel, surv)
    return screened.book(panel, rec, "inv_vol", freq)


def test_each_segment_starts_from_cash_and_a_segment_without_c_t_is_not_scored():
    book = _gapped_book()
    per = ex.holding_periods(book)
    res = runs.run(runs.prepare(book, filters.rolling), 1.0, per)
    first_forecast = pd.Period("1980-01", "M") + 119                 # the mix's 120th month (data from 1980)
    first_size = first_forecast + 120 + 1                            # c_t needs 120 pairs, used the month after
    assert res.index[0] == first_size and res.index[0].year == 2000  # 1991-94 and 1997-99 have no c_t yet: cash
    assert res["entry"].sum() == 1
    assert res["turnover"].iloc[0] == pytest.approx(res["size"].iloc[0] * book.positions.loc[res.index[0]].abs().sum(), rel=1e-12)
    m = runs.scored(res, per)
    assert np.allclose(m["excess"], res["excess"]) and np.allclose(m["bench_excess"], res["bench_excess"])


def test_a_weekly_run_compounds_its_weeks_into_calendar_months():
    book = _gapped_book(WEEK)
    per = ex.holding_periods(book)
    res = runs.run(runs.prepare(book, weekly.rolling), 1.0, per)
    m = runs.scored(res, per)
    month = runs.month_of(res, per)
    one = m.index[10]
    wk = res[np.asarray(month == one)]
    assert m.loc[one, "excess"] == pytest.approx(np.prod(1 + wk["net"]) - np.prod(1 + wk["rf"]), rel=1e-12)
    assert (per.closes.loc[wk.index].dt.to_period("M") == one).all()


def test_downside_is_the_22_day_downside_rv_under_rolling_rv(p):
    f = runs.downside(filters.rolling, MONTH)(p)
    o, _ = inputs.schedule(p)
    d = o.iloc[7]
    last = p.iloc[d - 21:d + 1]
    if (last < 0).sum() >= 3:
        assert f.iloc[7] == pytest.approx((last[last < 0] ** 2).sum(), rel=1e-10)
    q = pd.Series(np.r_[-0.01, -0.02, np.full(40, 0.01), -0.005, 0.01], index=pd.bdate_range("2000-01-03", periods=44))
    share = runs.down_share(q)
    assert share.iloc[-1] == pytest.approx((0.01 ** 2 + 0.02 ** 2 + 0.005 ** 2) / (q ** 2).sum())   # 44 days: one down day in 22


def test_break_even_closes_the_sharpe_gap_and_metrics_leave_out_the_entry_trade():
    book = _gapped_book()
    per = ex.holding_periods(book)
    res = runs.run(runs.prepare(book, filters.rolling), 1.0, per)
    out = runs.metrics(res, per)
    if out["gain"] > 0:
        k = out["break_even"]
        x = runs.excess_months(res, per, kappa=k)
        b = runs.excess_months(res, per, "bench_", kappa=k)
        assert ev.sharpe(x) == pytest.approx(ev.sharpe(b), abs=1e-8)
    assert out["turnover_per_rebalance"] == pytest.approx(res["turnover"].iloc[1:].mean())
    x = runs.scored(res, per)["excess"]
    assert out["sortino"] == pytest.approx(x.mean() / np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(12))


def test_the_etf_forecaster_is_phase_6s_walk_forward_with_a_shorter_start(p):
    assert np.array_equal(tracks.etf_garch(True, 120)(p).to_numpy(), filters.garch_fits(p, True)[0].to_numpy())
    f12 = tracks.etf_garch(False)(p)
    assert f12.index[0] == inputs.origins(p).index[11]


# ---------------------------------------------------------------- the tests

def test_bootstrap_counts_draw_n_months_in_blocks_of_the_given_mean_length():
    C = stats.counts(200, 12, reps=2000, seed=1)
    assert C.shape == (2000, 200) and (C.sum(axis=1) == 200).all()
    rng = np.random.default_rng(1)                               # the same draws, rebuilt: mean block length
    idx = np.empty((2000, 200), dtype=np.int64)
    idx[:, 0] = rng.integers(0, 200, 2000)
    jump = rng.random((2000, 200)) < 1 / 12
    fresh = rng.integers(0, 200, (2000, 200))
    for t in range(1, 200):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % 200)
    assert np.array_equal(np.bincount((np.arange(2000)[:, None] * 200 + idx).ravel(), minlength=2000 * 200).reshape(2000, 200), C)
    assert 1 / jump[:, 1:].mean() == pytest.approx(12, rel=0.05)


def test_bootstrap_sharpe_and_cer_match_the_direct_formulas():
    rng = np.random.default_rng(0)
    X = rng.normal(0.005, 0.04, (150, 3))
    C = stats.counts(150, 6, reps=5, seed=2)
    k = 3
    draw = np.repeat(np.arange(150), C[k])
    s = pd.Series(X[draw, 1])
    assert stats.sharpe_draws(C, X)[k, 1] == pytest.approx(ev.sharpe(s), rel=1e-10)
    assert stats.cer_draws(C, X, 5)[k, 1] == pytest.approx(ev.cer(s, 5), rel=1e-10)


def test_romano_wolf_matches_a_brute_force_stepdown_and_bounds_holm():
    rng = np.random.default_rng(0)
    t = np.array([3.5, 2.6, 2.4, 0.3, -1.0])
    draws = rng.standard_normal((4000, 5)) @ np.linalg.cholesky(0.5 * np.eye(5) + 0.5).T
    p = stats.romano_wolf(t, draws)
    order = np.argsort(-t)
    brute, running = np.empty(5), 0.0
    for j, k in enumerate(order):
        running = max(running, (draws[:, order[j:]].max(axis=1) >= t[k]).mean())
        brute[k] = running
    assert np.allclose(p, brute)
    raw = norm.sf(t)
    assert (p <= stats.holm(raw) + 0.02).all()                 # correlation makes Romano-Wolf less conservative


def test_romano_wolf_keeps_the_familywise_error_at_five_percent_when_nothing_works():
    rng = np.random.default_rng(5)
    false = 0
    for _ in range(300):
        x = rng.standard_normal((120, 8)) @ np.linalg.cholesky(0.6 * np.eye(8) + 0.4).T
        mu, se = x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(120)
        C = stats.counts(120, 6, reps=500, seed=int(rng.integers(1 << 30)))
        m1 = C @ x / 120
        false += (stats.romano_wolf(mu / se, (m1 - mu) / se) < 0.05).any()
    assert false / 300 < 0.09


def test_holm_matches_statsmodels():
    p = np.array([0.001, 0.04, 0.03, 0.2, 0.0004, 0.6])
    assert np.allclose(stats.holm(p), multipletests(p, method="holm")[1])
    assert stats.holm(np.array([0.05 / 192 * 0.99] + [0.5] * 191))[0] < 0.05


def test_deflated_sharpe_is_the_probabilistic_sharpe_at_the_selection_threshold():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_t(5, 400) * 0.03 + 0.012)
    sr = x.mean() / x.std()
    from scipy.stats import kurtosis, skew
    out = stats.deflated_sharpe(x, 0.2, 0.0004, 192)
    thr = 0.2 + 0.02 * stats.expected_max(192)
    z = (sr - thr) * np.sqrt(399) / np.sqrt(1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr ** 2)
    assert out["threshold_monthly"] == pytest.approx(thr) and out["dsr"] == pytest.approx(norm.cdf(z))
    assert stats.deflated_sharpe(x, 0.2, 0.0004, 1000)["dsr"] < out["dsr"] < stats.deflated_sharpe(x, 0.2, 0.0004, 10)["dsr"]
    assert 2.5 < stats.expected_max(192) < 2.9


def test_the_combination_follows_its_formulas_and_its_weight_limit():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0.006, 0.04, 300), pd.period_range("1990-01", periods=300, freq="M"))
    m = 0.6 * b + pd.Series(rng.normal(0.003, 0.01, 300), b.index)
    out = stats.combination(m, b)
    assert len(out) == 180 and out.index[0] == b.index[120]
    assert (out[["w_bench", "w_managed"]].abs().sum(axis=1) <= 5 + 1e-12).all()
    t = 150
    past = np.column_stack([b.iloc[:t], m.iloc[:t]])
    w = np.linalg.solve(np.cov(past, rowvar=False), past.mean(axis=0)) / 5
    w = w * min(1, 5 / np.abs(w).sum())
    row = out.loc[b.index[t]]
    assert np.allclose([row["w_bench"], row["w_managed"]], w)
    assert row["combined"] == pytest.approx(w @ [b.iloc[t], m.iloc[t]])
    assert row["w_alone"] == pytest.approx(np.clip(past[:, 0].mean() / (5 * past[:, 0].var(ddof=1)), -5, 5))


def test_the_bootstrap_p_value_is_small_for_a_clear_gain_and_not_for_none():
    rng = np.random.default_rng(0)
    C = stats.counts(300, 12, reps=3000, seed=0)
    gap = rng.normal(0.004, 0.01, 300)
    stat = np.array([gap.mean()])
    draws = (C @ gap / 300)[:, None]
    assert stats.bootstrap_p(stat, draws)[0] < 0.01
    null = rng.normal(0, 0.01, 300)
    assert stats.bootstrap_p(np.array([null.mean()]), (C @ null / 300)[:, None])[0] > 0.01 or abs(null.mean()) > 0.0015


# ---------------------------------------------------------------- option (c), Entry 10

def test_year_scale_is_one_over_the_books_volatility_over_its_recipe_window():
    panel = _panel()
    surv = _survivors(panel, {1992: ["mkt"], 1993: ["mkt", "bab"]})
    rec = screened.recipes(panel, surv)
    s = screened.year_scale(panel, rec, "inv_vol")
    assert list(s.index) == [1993, 1994]
    row = rec.loc[(pd.Timestamp("1993-12-31"), "inv_vol")]
    win = panel.loc[row["window_start"]:"1993-12-31", ["mkt", "bab"]]
    p = win @ row[["w_mkt", "w_bab"]].astype(float).to_numpy()
    assert s[1994] == pytest.approx(1 / (p.std() * np.sqrt(252)), rel=1e-12)
    assert s[1993] == pytest.approx(1 / (panel.loc[rec.loc[(pd.Timestamp("1992-12-31"), "inv_vol"), "window_start"]:"1992-12-31", "mkt"].std() * np.sqrt(252)), rel=1e-12)


def test_year_scale_uses_only_data_up_to_each_year_end():
    panel = _panel()
    surv = _survivors(panel, {1992: ["mkt", "bab"], 1993: ["mkt", "bab"]})
    noisy = panel.copy()
    noisy.loc["1993-01-01":, FACTORS] *= 5
    a = screened.year_scale(panel, screened.recipes(panel, surv), "inv_vol")
    b = screened.year_scale(noisy, screened.recipes(noisy, surv), "inv_vol")
    assert a[1993] == b[1993] and a[1994] != b[1994]


def test_year_weighting_scales_both_runs_alike_and_takes_the_weight_off_a_wild_year():
    rng = np.random.default_rng(0)
    months = pd.period_range("1990-01", "2001-12", freq=MONTH)
    vol = np.where(months.year == 1990, 0.10, 0.01)                    # one year ten times riskier
    b = pd.Series(rng.normal(0.004, 1, len(months)) * vol, months)
    m = pd.DataFrame({"excess": 0.7 * b + rng.normal(0.001, 0.3, len(months)) * vol, "bench_excess": b})
    flat = pd.Series(3.0, index=range(1990, 2002))
    same = runs.year_weighted(m, flat)
    assert np.allclose(same["excess"] / m["excess"], 3.0) and np.allclose(same["bench_excess"] / m["bench_excess"], 3.0)
    assert ev.sharpe_gain_test(same.assign(size=1.0))["gain"] == pytest.approx(ev.sharpe_gain_test(m.assign(size=1.0))["gain"], rel=1e-10)
    scale = pd.Series(np.where(np.arange(1990, 2002) == 1990, 10.0, 100.0), index=range(1990, 2002))   # 1 / ex-ante vol
    w = runs.year_weighted(m, scale)
    share = lambda x: (x[x.index.year == 1990] ** 2).sum() / (x ** 2).sum()
    assert share(m["bench_excess"]) > 0.7 and share(w["bench_excess"]) < 0.2
    with pytest.raises(ValueError, match="no scale"):
        runs.year_weighted(m, scale.drop(1995))
