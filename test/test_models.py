"""Phase 6: the forecast models and the scorecards, on simulated data only."""

import numpy as np
import pandas as pd
import pytest

from src import engine as e
from src.engine import evaluation as ev, synthetic as syn
from src.models import ORDER, boost, filters, forecasters, har, inputs, oracle
from src.models import scorecard as sc

YEARS = 14


def _p(years=YEARS, seed=0, start="1990-01-01"):
    return syn.garch(years=years, seed=seed, start=start)["x"].rename("p")


def _panel(index, seed=0):
    """A fake panel for the exogenous inputs: VIX from year 6, events from year 7, so both enter
    (60 months later) inside the forecast years 10-14."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(index=index)
    out["vix"] = pd.Series(15 + 5 * rng.random(len(index)), index).where(index >= index[0] + pd.DateOffset(years=6))
    for ev_name in ["fomc", "cpi", "jobs"]:
        flags = pd.Series(rng.random(len(index)) < 0.05, index).astype("boolean")
        out[ev_name] = flags.where(index >= index[0] + pd.DateOffset(years=7))
    return out


@pytest.fixture(scope="module")
def p():
    return _p()


@pytest.fixture(scope="module")
def fs(p):
    return forecasters(_panel(p.index), down=True)


# ---------------------------------------------------------------- inputs

def test_qlike_is_zero_when_perfect_and_punishes_under_forecasts_harder():
    assert inputs.qlike(2.0, 2.0) == 0
    assert inputs.qlike(10.0, 1.0) == pytest.approx(10 - np.log(10) - 1)        # ratio 10: 6.70
    assert inputs.qlike(0.1, 1.0) == pytest.approx(0.1 + np.log(10) - 1)        # ratio 0.1: 1.40
    assert inputs.qlike(10.0, 1.0) > inputs.qlike(0.1, 1.0)


def test_first_forecast_is_at_the_end_of_the_120th_month_and_yearly_refits_are_decembers(p):
    o, refit = inputs.schedule(p, "month")
    months = inputs.origins(p).index
    assert o.index[0] == months[inputs.MIN_MONTHS - 1] and refit.all()
    assert p.index[o.iloc[0]] == p.loc[str(o.index[0])].index[-1]              # the month's last trading day
    o, refit = inputs.schedule(p, "year")
    assert refit[0] and set(o.index[refit][1:].month) == {12}


def test_forward_sums_the_next_22_days_and_target_reads_it_at_month_ends(p):
    r2 = p ** 2
    f = inputs.forward(r2)
    d = 500
    assert f.iloc[d] == pytest.approx(r2.iloc[d + 1:d + 23].sum())
    assert f.iloc[-22:].isna().all() and f.iloc[-23:-22].notna().all()
    y = inputs.target(p)
    pos = inputs.origins(p).iloc[40]
    assert y.iloc[40] == pytest.approx(r2.iloc[pos + 1:pos + 23].sum())


def test_har_design_floors_logs_and_down_shares_split_the_lags(p):
    X = inputs.har_design(p, down=True)
    r2 = p ** 2
    m = r2.rolling(22).mean()
    d = 300
    assert X["m"].iloc[d] == pytest.approx(np.log(m.iloc[d]))
    assert X["d"].iloc[d] == pytest.approx(np.log(max(r2.iloc[d], inputs.FLOOR * m.iloc[d])))
    assert (X["d"].dropna() >= np.log(inputs.FLOOR * m).dropna().reindex(X["d"].dropna().index) - 1e-12).all()
    assert ((X["d_down"] == 1) == (p < 0)).iloc[30:].all()
    w = p.iloc[d - 4:d + 1]
    assert X["w_down"].iloc[d] == pytest.approx((w[w < 0] ** 2).sum() / (w ** 2).sum())


def test_event_counts_look_22_days_ahead_and_are_missing_before_the_calendar(p):
    panel = _panel(p.index)
    x = inputs.exog(panel, p.index)
    flags = panel["cpi"].astype(float)
    d = 2000
    assert x["cpi"].iloc[d] == flags.iloc[d + 1:d + 23].sum()
    start = p.index.get_loc(flags.first_valid_index())
    assert x["cpi"].iloc[:start - 1].isna().all() and x["cpi"].iloc[start - 1:].notna().all()
    assert x["vix"].dropna().iloc[0] == pytest.approx(np.log((panel["vix"].dropna().iloc[0] / 100) ** 2 / 12))


# ---------------------------------------------------------------- M0, M1

def test_rolling_and_ewma_measure_what_they_say(p):
    o, _ = inputs.schedule(p)
    month, pos = o.index[5], o.iloc[5]
    r2 = (p ** 2).to_numpy()
    assert filters.rolling(p)[month] == pytest.approx(r2[pos - 21:pos + 1].sum())
    s = r2[0]
    for x in r2[1:pos + 1]:                                       # σ̂²_{d+1} = λ σ̂²_d + (1 - λ) r²_d
        s = filters.LAMBDA * s + (1 - filters.LAMBDA) * x
    assert filters.ewma(p)[month] == pytest.approx(22 * s, rel=1e-6)


# ---------------------------------------------------------------- M2, M3

def test_garch_recovers_the_simulated_parameters_and_gjr_finds_no_asymmetry():
    r = _p(years=80, seed=3).to_numpy() * filters.SCALE
    r2, neg = r ** 2, np.where(r < 0, r ** 2, 0.0)
    theta, ok = filters.garch_fit(r2, neg, r2.mean(), gjr=False)
    assert ok and theta[1] == pytest.approx(0.10, abs=0.01) and theta[3] == pytest.approx(0.89, abs=0.01)
    theta, ok = filters.garch_fit(r2, neg, r2.mean(), gjr=True)
    assert ok and abs(theta[2]) < 0.02


@pytest.mark.parametrize("gjr", [False, True])
def test_a_garch_refit_never_ends_worse_than_its_start_and_keeps_omega_in_scale(gjr):
    """The first real-data run had SLSQP report success at a likelihood far worse than its warm
    start, with ω 10⁹ times too large (PREREGISTRATION Entry 5). Low-volatility returns, like the
    combined mix's, and starts both good and poor."""
    r = 0.3 * _p(years=30, seed=5).to_numpy() * filters.SCALE
    r2, neg = r ** 2, np.where(r < 0, r ** 2, 0.0)
    nll = lambda t: filters._nll(t, r2, neg, r2.mean())
    good, _ = filters.garch_fit(r2, neg, r2.mean(), gjr)
    for start in (good, np.array([0.5 * r2.mean(), 0.01, 0.0, 0.01]), np.array([1e-4 * r2.mean(), 0.3, 0.0, 0.69])):
        theta, _ = filters.garch_fit(r2, neg, r2.mean(), gjr, start)
        assert nll(theta) <= nll(np.minimum(start, [r2.mean(), 1, 1, 1])) + 1e-9
        assert theta[0] <= r2.mean() and filters.persistence(theta) < 1
    assert good[3] > 0.8                                          # the simulated b is 0.89


def test_garch_h_step_sum_matches_iterating_the_recursion():
    theta = np.array([0.02, 0.07, 0.06, 0.88])
    s2_next, total, s = 1.7, 0.0, 1.7
    rho, w = filters.persistence(theta), theta[0]
    for _ in range(22):
        total += s
        s = w + rho * s                                           # E[σ²_{t+h+1}] = ω + ρ E[σ²_{t+h}]
    assert filters.garch_sum(theta, s2_next) == pytest.approx(total)


# ---------------------------------------------------------------- M4-M6

def test_har_forecast_is_the_bias_corrected_ols_fit_on_rows_whose_target_is_known(p):
    f, fits = har.har_fits(p)
    o, _ = inputs.schedule(p)
    k = 12
    month, pos = o.index[k], o.iloc[k]
    X = inputs.har_design(p)
    y = np.log(inputs.forward(p ** 2))
    train = pd.concat([X, y.rename("y")], axis=1).iloc[:pos - 22 + 1].dropna()   # targets end by the origin
    A = np.column_stack([np.ones(len(train)), train[["d", "w", "m"]]])
    beta = np.linalg.lstsq(A, train["y"], rcond=None)[0]
    s2 = np.sum((train["y"] - A @ beta) ** 2) / (len(train) - 4)
    assert fits.loc[month, "rows"] == len(train)
    assert f[month] == pytest.approx(np.exp(beta[0] + X.iloc[pos].to_numpy() @ beta[1:] + s2 / 2))


def test_vix_and_events_enter_har_x_after_60_months_of_history(p):
    x = inputs.exog(_panel(p.index), p.index)
    _, fits = har.har_fits(p, down=False, x=x)
    vix_from = x["vix"].first_valid_index() + pd.DateOffset(months=inputs.X_MIN_MONTHS)
    used = fits["vix"].notna()
    assert not used.iloc[0] and used.iloc[-1]
    first = fits.index[used][0]
    assert first.to_timestamp(how="end") >= vix_from
    m4, _ = har.har_fits(p)
    m6, _ = har.har_fits(p, down=False, x=x)
    assert np.allclose(m4[~used.to_numpy()], m6[~used.to_numpy()])       # before that, M6 is M4


# ---------------------------------------------------------------- every model

@pytest.mark.parametrize("name", ORDER)
def test_forecasts_use_only_data_up_to_their_origin(p, name):
    cut = p.index[int(len(p) * 0.93)]
    noisy = p.copy()
    noisy[noisy.index > cut] = np.random.default_rng(1).normal(0, 0.05, (noisy.index > cut).sum())
    panel = _panel(p.index)
    a = forecasters(panel, down=True)[name](p)
    b = forecasters(panel, down=True)[name](noisy)
    before = a.index[a.index.to_timestamp(how="end") <= cut]
    assert len(before) > 20 and a.loc[before].notna().all()
    pd.testing.assert_series_equal(a.loc[before], b.loc[before])
    assert (a.dropna() > 0).all()


def test_m7_is_tuned_inside_its_training_rows_and_refit_yearly(p):
    x = inputs.exog(_panel(p.index), p.index)
    f, fits = boost.boost_fits(p, True, x)
    assert set(fits.index[1:].month) == {12} and fits.index[0] == f.index[0]
    assert set(fits["leaves"]) <= set(boost.LEAVES) and set(fits["trees"]) <= set(boost.TREES)
    assert fits["rows"].is_monotonic_increasing


def test_every_model_plugs_into_the_engine_and_starts_on_the_same_month(p, fs):
    p = p.loc[:"2003-06"]                                          # whole months: the oracle needs the last one
    book = e.Book.single(p, pd.Series(0.0001, p.index))
    runs = {m: sc.run(book, f, e.NO_COSTS, min_months=24) for m, f in fs.items()}   # 14 years: a short c_t window
    first = {r.index[0] for r in runs.values()}
    assert len(first) == 1
    sc._common_index(runs)
    assert oracle.looks_ahead and sc.run(book, oracle, e.NO_COSTS, min_months=24).index[0] == runs["M0"].index[0]


# ---------------------------------------------------------------- scorecards

def test_targets_follow_the_recipe_held_and_the_decision_month():
    a, b = _p(seed=1), _p(seed=2)
    months = a.index.to_period("M").unique()
    pos = pd.DataFrame({"A": np.where(months.year % 2 == 0, 1.0, 0.3), "B": 0.5}, index=months)
    book = e.Book(pd.DataFrame({"A": a, "B": b}), pd.Series(0.0, a.index), pos)
    y = sc.targets(book)
    m = months[30]
    mix = book.returns @ pos.loc[m]
    assert y[m] == pytest.approx(inputs.target(mix)[m - 1])


def test_mcs_drops_a_clearly_worse_model_and_keeps_ties():
    rng = np.random.default_rng(0)
    base = rng.gamma(2.0, 0.5, 600)
    L = pd.DataFrame({"a": base + rng.normal(0, 0.05, 600), "b": base + rng.normal(0, 0.05, 600),
                      "bad": base + 0.5 + rng.normal(0, 0.05, 600)})
    res = sc.mcs(L, reps=2000)
    assert res.loc["bad", "mcs_p"] < 0.10 and not res.loc["bad", "in_mcs"]
    assert res.loc[["a", "b"], "in_mcs"].all()
    ties = sc.mcs(L[["a", "b"]], reps=2000)
    assert ties["in_mcs"].all()


def test_mcs_p_values_do_not_fall_as_models_are_dropped():
    rng = np.random.default_rng(1)
    L = pd.DataFrame({f"m{k}": rng.gamma(2.0, 0.5, 400) + 0.08 * k for k in range(6)})
    res = sc.mcs(L, reps=2000).sort_values("dropped_in_round")
    assert res["mcs_p"].is_monotonic_increasing and res["mcs_p"].iloc[-1] == 1.0
    assert (res["in_mcs"] == (res["mcs_p"] >= 0.10)).all()
    assert res.loc["m0", "in_mcs"] and not res.loc["m5", "in_mcs"]


def _fake_runs(sharpes, seed=0, n=600):
    """Runs whose excess returns share one driver, so their Sharpe gaps have small errors."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.02, n)
    out = {}
    for m, s in zip(ORDER, sharpes):
        x = common + rng.normal(0, 0.004, n)
        out[m] = pd.DataFrame({"excess": x - x.mean() + s * x.std() / np.sqrt(12)},
                              index=pd.period_range("1950-01", periods=n, freq="M"))
    return out


def test_selection_takes_the_simplest_model_in_the_mcs_within_one_error_of_the_best():
    runs = _fake_runs([0.50, 0.90, 0.95, 0.96, 0.97, 0.99, 1.00, 0.98])
    a = pd.DataFrame({"in_mcs": [True, False, True, True, True, True, True, True]}, index=ORDER)
    t = sc.select(a, runs)
    assert t.loc["M6", "best"] and not t.loc["M0", "within_1se"]
    pick = t.index[t["chosen"]][0]
    first = next(m for m in ORDER if t.loc[m, "within_1se"] and t.loc[m, "in_mcs"])
    assert pick == first and not t["fallback"].any()


def test_selection_drops_the_mcs_condition_when_no_member_is_within_one_error():
    runs = _fake_runs([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.20])
    a = pd.DataFrame({"in_mcs": [True] * 7 + [False]}, index=ORDER)
    t = sc.select(a, runs)
    assert t["fallback"].all() and t.loc["M7", "chosen"]


def test_sharpe_difference_matches_the_primary_test_and_cer_its_formula(p):
    book = e.Book.single(p, pd.Series(0.0001, p.index))
    res = e.run(book, min_months=24)
    gain, se = ev.sharpe_difference(res["excess"], res["bench_excess"])
    test = ev.sharpe_gain_test(res)
    assert gain == test["gain"] and se == test["se"]
    x = res["excess"]
    assert ev.cer(x, 5) == pytest.approx(12 * x.mean() - 2.5 * 12 * x.var())
