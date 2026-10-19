"""Phase 8: daily holding periods, the GARCH variations, time-each-then-combine, the daily crisis paths
and the scoring helpers, on simulated data only."""

import numpy as np
import pandas as pd
import pytest

from src import engine as e
from src import portfolio as pf
from src.config import FACTORS
from src.engine import data, evaluation as ev, execution as ex, synthetic as syn, weights
from src.engine.data import DAY, MONTH, WEEK
from src.models import filters, inputs, variations
from src.robustness import CHECKS, checks, crises
from src.walkforward import runs, screened


def _p(years=14, seed=0):
    return syn.garch(years=years, seed=seed, start="1990-01-01")["x"].rename("p")


@pytest.fixture(scope="module")
def p():
    return _p()


def _rf(x):
    return pd.Series(0.0001, x.index)


def _panel(years=24, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("1980-01-01", periods=252 * years, name="date")
    panel = pd.DataFrame(rng.normal(0.0003, 0.01, (len(idx), len(FACTORS))), index=idx, columns=FACTORS)
    panel["rf"] = 0.0001
    return panel


def _daily_forecaster(values):
    """A forecaster giving each trading day the value of `values` (a function of the daily returns)."""
    def f(x):
        x = x.dropna()
        out = values(x)
        out.index = x.index.to_period(DAY)
        return out.rename("forecast")
    return f


# ---------------------------------------------------------------- engine: trading days

def test_shift_steps_along_trading_days_and_leaves_months_and_weeks_alone():
    cal = pd.bdate_range("2000-01-03", periods=10).to_period(DAY)
    fri, mon = pd.Period("2000-01-07", DAY), pd.Period("2000-01-10", DAY)
    assert data.shift(pd.PeriodIndex([mon]), -1, cal)[0] == fri
    assert data.shift(pd.PeriodIndex([fri]), 1, cal)[0] == mon
    assert data.shift(cal[-1:], 1, cal).isna().all() and data.shift(cal[:1], -1, cal).isna().all()
    months = pd.period_range("2000-01", periods=4, freq=MONTH)
    assert (data.shift(months, -1, cal) == months - 1).all()
    weeks = pd.period_range("2000-01-07", periods=4, freq=WEEK)
    assert (data.shift(weeks, 1, cal) == weeks + 1).all()


def test_a_daily_book_earns_the_asset_each_day_and_a_weekend_is_no_gap(p):
    book = e.Book.single(p, _rf(p), "A", freq=DAY)
    per = ex.holding_periods(book)
    days = per.growth.index[100:140]
    out = ex.simulate(per, pd.DataFrame({"A": 1.0}, index=days))
    assert np.allclose(out["net"], per.growth.loc[days, "A"] - 1)
    assert np.allclose(out["excess"], p.loc[days.to_timestamp()].to_numpy(), atol=1e-15)
    assert (per.opens.loc[days[1:]].to_numpy() == days[:-1].to_timestamp().to_numpy()).all()
    with pytest.raises(ValueError, match="consecutive"):
        ex.simulate(per, pd.DataFrame({"A": 1.0}, index=days.delete(5)))
    tab = data.table(p, _rf(p), DAY)
    assert np.allclose(tab["ret"].to_numpy(), p.to_numpy()) and np.allclose(tab["rv"].to_numpy(), p.to_numpy() ** 2)
    assert len(runs.segments(book)) == 1


def test_a_daily_book_holds_trading_days_only(p):
    with pytest.raises(ValueError, match="trading days"):
        e.Book(p.to_frame("A"), _rf(p), pd.DataFrame({"A": 1.0}, index=pd.period_range("1990-01-01", periods=10, freq=DAY)))


def test_daily_c_t_pairs_each_signal_with_the_next_trading_days_return(p):
    rf = _rf(p)
    tab = data.table(p, rf, DAY)
    sig = weights.signal(_daily_forecaster(lambda x: x.rolling(22).var() * 22)(p), 1.0)
    c = weights.realtime_c(tab["ret"], sig, min_months=300)
    t = c.index[500]
    pos = tab.index.get_loc(t)
    y = tab["ret"].iloc[1:pos + 1].to_numpy()                   # returns of days 1..t, each the day after a signal
    z = y * sig.reindex(tab.index[:pos]).to_numpy()
    ok = ~np.isnan(z)
    assert c[t] == pytest.approx(np.std(y[ok], ddof=1) / np.std(z[ok], ddof=1), rel=1e-12)


def test_a_daily_size_uses_the_forecast_made_at_the_close_before(p):
    book = e.Book.single(p, _rf(p), "A", freq=DAY)
    f = _daily_forecaster(lambda x: (x ** 2).rolling(22).sum())
    dec = e.decide(book, e.prepare(book, f), 1.0, min_months=300)
    fc = f(p)
    monday = next(d for d in dec.index if d.to_timestamp().dayofweek == 0)
    assert dec.loc[monday, "forecast"] == fc[monday - 3]                 # the Friday before
    flat = e.decide(book, e.prepare(book, _daily_forecaster(lambda x: pd.Series(2.0, x.index))), 1.0, min_months=300)
    assert np.allclose(flat["size"], 1.0, atol=1e-12)


def test_a_daily_screened_book_switches_recipes_on_the_first_day_after_the_year_end():
    panel = _panel()
    surv = pd.Series({d: (["mkt", "bab"] if d.year == 1993 else ["mkt"] if d.year == 1992 else [])
                      for d in screened.sc.year_ends(panel.index)})
    rec = screened.recipes(panel, surv)
    book = screened.book(panel, rec, "inv_vol", DAY)
    days = book.positions.index.to_timestamp()
    assert days[0].year == 1993 and days[-1].year == 1994 and days[0] == panel.loc["1993"].index[0]
    assert (book.positions.loc[days.year == 1993, "bab"] == 0).all() and (book.positions.loc[days.year == 1994, "bab"] > 0).all()
    assert len(runs.segments(book)) == 1


# ---------------------------------------------------------------- the GARCH variations

@pytest.mark.parametrize("gjr", [False, True])
def test_the_daily_garch_forecast_is_the_first_term_of_the_monthly_sum(p, gjr):
    f, params = filters.garch_fits(p, gjr)
    d = variations.garch_daily(p, gjr, params)
    mo, _ = inputs.schedule(p)
    days, _ = variations.daily_schedule(p)
    assert days.iloc[0] == mo.iloc[0] and (days.diff().dropna() == 1).all()
    for month, R in list(mo.items())[::7]:
        theta = params.loc[month, ["omega", "a", "g", "b"]].to_numpy()
        day = p.index[R].to_period(DAY)
        monthly = filters.garch_sum(theta, d[day] * filters.SCALE ** 2) / filters.SCALE ** 2
        assert monthly == pytest.approx(f[month], rel=1e-12)


def test_the_daily_garch_forecast_uses_only_data_up_to_its_day(p):
    cut = p.index[int(len(p) * 0.8)]
    q = p.copy()
    q[q.index > cut] = np.random.default_rng(1).normal(0, 0.05, (q.index > cut).sum())
    a = variations.garch_daily(p, True, filters.garch_fits(p, True)[1])
    b = variations.garch_daily(q, True, filters.garch_fits(q, True)[1])
    known = a.index[a.index.to_timestamp() <= cut]
    assert len(known) > 200 and np.array_equal(a[known].to_numpy(), b[known].to_numpy())


def test_a_window_longer_than_the_data_is_phase_6s_expanding_walk_forward(p):
    f, params = filters.garch_fits(p, True)
    g, gparams = variations.garch_fits_window(p, True, 10 ** 6)
    assert np.array_equal(f.to_numpy(), g.to_numpy())
    assert np.array_equal(params[["omega", "a", "g", "b"]].to_numpy(), gparams[["omega", "a", "g", "b"]].to_numpy())


def test_a_rolling_window_fits_on_its_last_days_only_and_never_looks_ahead(p):
    days = 2000
    g, params = variations.garch_fits_window(p, False, days)
    o, _ = inputs.schedule(p)
    assert (params["window_start"].to_numpy() == p.index[np.maximum(o.to_numpy() + 1 - days, 0)].to_numpy()).all()
    cut = p.index[o.iloc[30]]
    q = p.copy()
    q[q.index > cut] = q[q.index > cut] * 3
    h, _ = variations.garch_fits_window(q, False, days)
    assert np.array_equal(g.iloc[:31].to_numpy(), h.iloc[:31].to_numpy()) and not np.allclose(g.iloc[31:], h.iloc[31:])


# ---------------------------------------------------------------- time each, then combine

def test_timing_one_asset_alone_is_timing_the_mix():
    x = _p(24)
    book = e.Book.single(x, _rf(x), "A")
    per = ex.holding_periods(book)
    a = runs.run(runs.prepare(book, filters.rolling), 1.0, per)
    b = checks.time_each(book, filters.rolling, per)
    assert a.index.equals(b.index)
    for col in ("size", "excess", "bench_excess", "turnover", "entry"):
        assert np.allclose(a[col].astype(float), b[col].astype(float), rtol=0, atol=1e-14)


def test_time_each_gives_each_asset_its_own_size():
    panel = _panel(26)
    panel["mom"] = panel["mom"] * np.where(panel.index.year % 2, 2.0, 0.5)    # momentum's risk moves; the market's does not
    surv = pd.Series({d: ["mkt", "mom"] for d in screened.sc.year_ends(panel.index)})
    rec = screened.recipes(panel, surv)
    book = screened.book(panel, rec, "inv_vol")
    per = ex.holding_periods(book)
    res = checks.time_each(book, filters.rolling, per)
    for a in ("mkt", "mom"):
        single = e.Book.single(panel[a], panel["rf"], a)
        size = e.decide(single, e.prepare(single, filters.rolling), 1.0)["size"]
        m = res.index[20]
        assert res.loc[m, f"w_{a}"] == pytest.approx(book.positions.loc[m, a] * size[m], rel=1e-12)
    assert res["size"].iloc[20] == pytest.approx(res.loc[res.index[20], ["w_mkt", "w_mom"]].sum(), rel=1e-12)
    assert np.allclose(res["bench_w_mkt"], book.positions.loc[res.index, "mkt"])


# ---------------------------------------------------------------- books and changes

def test_the_fixed_family_is_phase_4s_paper_recipe_and_drop_bab_its_market_momentum_twin():
    panel = _panel(26)
    rec = checks.recipes(panel, "fixed")
    pd.testing.assert_frame_equal(rec, pf.recipes(panel, "paper"))
    twin = checks.recipes(panel, "fixed", drop="bab")
    w = twin.filter(regex="^w_")
    assert np.allclose(w.to_numpy(), pf.recipes(panel, "paper_no_bab").filter(regex="^w_").to_numpy())
    mon = checks.book(panel, rec)
    old = e.data.from_phase4(panel, rec, "inv_vol")
    pd.testing.assert_frame_equal(mon.positions, old.positions, check_names=False)


def test_the_screened_family_drops_an_asset_from_every_year_end_and_leaves_empty_years_empty():
    panel = _panel()
    surv = pd.Series({d: (["mkt"] if d.year == 1992 else ["mkt", "bab"] if d.year > 1992 else [])
                      for d in screened.sc.year_ends(panel.index)})
    rec = checks.recipes(panel, "screened", surv, drop="mkt")
    years = sorted({d.year for d in rec.index.get_level_values("asof")})
    assert years[0] == 1993 and (rec["w_mkt"] == 0).all()
    assert checks.held_assets(checks.recipes(panel, "screened", surv)) == ["mkt", "bab"]


def test_every_check_lists_its_base_first_and_one_change_per_variation():
    v = checks.variations(["mkt", "mom"])
    assert [label for c, label, _ in v if c == "leave_one_out"] == ["none dropped", "drop market", "drop momentum"]
    bases = [(c, label) for c, label, change in v if not change]
    assert [c for c, _ in bases] == list(CHECKS)
    assert all(len(change) <= 2 for _, _, change in v)
    assert ("costs", "30 bp + borrowing at rf + 0.5%", {"kappa": 30e-4, "borrow": 0.005}) in v
    book = e.Book.single(_p(), _rf(_p()), "A")
    c = checks.costs_for(book, {"kappa": 20e-4})
    assert c.kappa == {"A": 20e-4} and c.borrow_spread == 0 and checks.costs_for(book, {}) == e.NO_COSTS


def test_a_daily_run_starting_inside_a_month_does_not_score_that_month(p):
    book = e.Book.single(p, _rf(p), "A", freq=DAY)
    res = pd.DataFrame(index=book.positions.index[15:400])          # holds from the 16th trading day
    months = checks.whole_months(book, res)
    first = book.positions.index[15].asfreq(MONTH)
    assert months[0] == first + 1 and first not in months
    mon = e.Book.single(p, _rf(p), "A")
    assert checks.whole_months(mon, pd.DataFrame(index=mon.positions.index[3:9])).equals(mon.positions.index[3:9])


def test_verdict_reads_signs_first_then_readings():
    assert checks.verdict(0.3, "edge", [0.2, 0.4], ["edge", "edge"]) == "holds"
    assert checks.verdict(0.3, "edge", [0.2, 0.05], ["edge", "inconclusive"]) == "sign holds"
    assert checks.verdict(0.01, "inconclusive", [0.02, -0.01], ["inconclusive"] * 2) == "flips"


def test_risk_shares_sum_to_one_and_follow_the_weights():
    panel = _panel(26)
    panel["mom"] *= 3
    rec = pf.recipes(panel, "paper")
    eq, iv = checks.risk_shares(panel, rec, "equal"), checks.risk_shares(panel, rec, "inv_vol")
    assert np.allclose(eq.sum(axis=1), 1) and np.allclose(iv.sum(axis=1), 1)
    assert (eq["mom"] > 0.75).all() and (iv["mom"] < eq["mom"]).all()


def test_placebo_shares_count_the_shuffled_draws_the_real_run_beats():
    months = pd.period_range("1990-01", periods=200, freq=MONTH)
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0.005, 0.03, 200), months)
    real = pd.DataFrame({"excess": 0.5 * b + 0.002, "bench_excess": b})
    draws = pd.DataFrame([0.5 * b + rng.normal(0, 0.004, 200) for _ in range(50)])
    out = checks.placebo_shares(draws, real, months)
    alphas = [ev.alpha(draws.loc[i], b) for i in draws.index]
    assert out["alpha_share"] == pytest.approx(np.mean(np.array(alphas) < ev.alpha(real["excess"], b)))
    assert out["pass"] == (out["alpha_share"] >= 0.95)


# ---------------------------------------------------------------- crises

def test_the_daily_path_compounds_to_the_engines_monthly_returns_with_costs_and_shorts():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("1990-01-01", periods=252 * 3)
    x = pd.DataFrame({"A": rng.normal(0.0004, 0.01, len(idx)), "B": rng.normal(0.0, 0.01, len(idx))}, index=idx)
    months = pd.period_range("1990-02", "1992-11", freq=MONTH)
    pos = pd.DataFrame({"A": 1.2, "B": -0.3}, index=months)            # levered at sizes above 1.1: borrowing
    book = e.Book(x, pd.Series(0.0001, idx), pos)
    costs = e.Costs(kappa={"A": 1e-3, "B": 2e-3}, borrow_spread=0.005, short_fee={"B": 0.01})
    per = ex.holding_periods(book, costs)
    dec = pd.DataFrame({"size": np.linspace(0.5, 1.8, len(months)), "forecast": 1.0}, index=months)
    res = e.trade(book, dec, per)
    path = crises.path(book, res, per)
    ends = path.groupby(path.index.to_period(MONTH)).tail(1)
    cash = np.cumprod(1 + book.rf.loc[path.index]).loc[ends.index].to_numpy()
    for pre in ("", "bench_"):
        assert np.allclose(ends[f"{pre}value"].to_numpy() * cash, np.cumprod(1 + res[f"{pre}net"]).to_numpy(), rtol=1e-12)
    assert path.loc["1991-03", "size"].nunique() == 1 and path.loc["1991-03-15", "size"] == res.loc["1991-03", "size"]


def test_a_span_runs_from_the_first_years_peak_to_the_lowest_close_after_it():
    idx = pd.bdate_range("2006-01-02", "2010-12-31")
    t = np.arange(len(idx))
    level = np.exp(np.where(idx.year < 2008, 0.0004 * t, 0.0004 * t[idx.year < 2008][-1] - 0.002 * (t - t[idx.year < 2008][-1])))
    level[idx >= "2009-03-02"] = level[idx < "2009-03-02"][-1] * np.exp(0.001 * np.arange((idx >= "2009-03-02").sum()))
    mkt = pd.Series(level, idx).pct_change().fillna(0.0)
    panel = pd.DataFrame({"mkt": mkt, "rf": 0.0})
    s = crises.span(panel, "2007", "2009")
    assert s["peak"] == idx[idx.year == 2007][-1] and s["trough"] == idx[idx < "2009-03-02"][-1]
    assert s["months"][0] == pd.Period("2007-06", MONTH) and s["months"][-1] == pd.Period("2010-02", MONTH)
