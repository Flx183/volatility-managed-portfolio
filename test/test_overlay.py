"""Track A: the variants' sizes, the Sortino gain test and the risk numbers, on simulated data only. No Track A
variant has been run on real data (PREREGISTRATION Entry 16)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from src import engine as e
from src.engine import evaluation as ev, execution as ex, forecast, synthetic as syn
from src.frozen import Frozen
from src.overlay import OPEN, TRACK_A, metrics, require_decided, sizing
from src.walkforward import runs, stats

PAIRS = 60                                            # c_t's minimum here: short simulated histories


@pytest.fixture(scope="module")
def setup():
    d = syn.garch(years=20, world="vol", seed=1, start="1990-01-01")
    book = e.Book.single(d["x"], d["rf"], "x")
    prep = runs.prepare(book, forecast.last_rv)
    return book, prep


def _run(setup, variant, costs=e.NO_COSTS, **kw):
    book, prep = setup
    return sizing.run(prep, ex.holding_periods(book, costs), variant, min_pairs=PAIRS, **kw)


def test_a0_is_the_frozen_runner_to_the_last_digit(setup):
    book, prep = setup
    pd.testing.assert_frame_equal(_run(setup, "A0"), runs.run(prep, 1.0, ex.holding_periods(book), min_pairs=PAIRS))


def test_a1_targets_the_books_own_volatility_from_its_last_year_end(setup):
    book, _ = setup
    sigma = sizing.sigma_trailing(book)
    a0, a1 = _run(setup, "A0"), _run(setup, "A1", sigma=sigma)
    assert a1.index.equals(a0.index)                                  # scored on A0's months
    target = sigma.reindex(a1.index.year).to_numpy() * np.sqrt(22 / 252)
    np.testing.assert_allclose(a1["size"], np.minimum(target / np.sqrt(a1["forecast"]), 1.5))
    np.testing.assert_allclose(a1["forecast"], a0["forecast"])        # the same forecast, another size


def test_a1_sizes_one_when_the_forecast_equals_the_target_and_caps_at_one_and_a_half():
    months = pd.period_range("2001-01", periods=3, freq="M")
    dec = pd.DataFrame({"forecast": [0.01, 0.0025, 1e-6], "signal": 1.0, "c": 1.0, "size": 1.0, "capped": False},
                       index=months)
    sigma = pd.Series({2001: np.sqrt(0.01 * 252 / 22)})              # a 22-day variance of 0.01
    out = sizing.resize(dec, "A1", sigma)
    np.testing.assert_allclose(out["size"], [1.0, 1.5, 1.5])
    assert out["capped"].tolist() == [False, True, True]
    with pytest.raises(ValueError, match="no σ̂_y"):
        sizing.resize(dec, "A1", pd.Series({2002: 0.1}))
    with pytest.raises(ValueError, match="unknown variant"):
        sizing.resize(dec, "A3")


def test_sigma_trailing_reads_ten_years_up_to_each_december_and_nothing_after(setup):
    book, _ = setup
    x = book.returns["x"]
    s = sizing.sigma_trailing(book)
    d = x.loc["2000-12"].index[-1]
    win = x.loc[(x.index > d - pd.DateOffset(months=120)) & (x.index <= d)]
    assert s[2001] == pytest.approx(win.std() * np.sqrt(252))
    later = x.copy()
    later.loc["2001-01-01":] *= 3                                      # the future changes; 2001's σ̂ does not
    s2 = sizing.sigma_trailing(e.Book.single(later, book.rf, "x"))
    assert s2[2001] == s[2001] and s2[2003] != s[2003]
    moving = e.Book(book.returns, book.rf, book.positions.assign(x=np.linspace(0.5, 1.0, len(book.positions))))
    with pytest.raises(ValueError, match="fixed positions"):
        sizing.sigma_trailing(moving)


def test_the_sortino_difference_and_its_standard_error_match_a_simulation():
    rng = np.random.default_rng(0)
    gains, ses = [], []
    for _ in range(2000):
        z = rng.standard_normal((240, 2))
        b = pd.Series(0.006 + 0.045 * z[:, 0])
        a = pd.Series(0.005 + 0.03 * (0.8 * z[:, 0] + 0.6 * z[:, 1]))
        g, se = metrics.sortino_difference(a, b, lags=0)
        gains.append(g)
        ses.append(se)
    assert g == pytest.approx(stats.sortino(a) - stats.sortino(b))
    assert np.std(gains) == pytest.approx(np.mean(ses), rel=0.08)


def test_the_sortino_gain_test_reads_like_the_primary_test():
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0.005, 0.04, 240))
    m = lambda a, bb: pd.DataFrame({"excess": a, "bench_excess": bb})
    up = metrics.sortino_gain_test(m(b + 0.01, b), meaningful=0.14)
    down = metrics.sortino_gain_test(m(b - 0.01, b), meaningful=0.14)
    few = b.iloc[:24]
    short = metrics.sortino_gain_test(m(pd.Series(few.sample(frac=1, random_state=0).to_numpy()), few), meaningful=0.14)
    assert short["gain"] == pytest.approx(0.0, abs=1e-12)              # the same months reordered: no gain, wide error
    assert (up["reading"], down["reading"], short["reading"]) == ("edge", "no meaningful edge", "inconclusive")
    assert up["mde"] == pytest.approx(2.4865 * up["se"], rel=1e-4) and up["months"] == 240


def test_expected_shortfall_is_the_mean_loss_of_the_worst_five_percent_of_months():
    x = pd.Series(np.arange(91, dtype=float) - 10)                    # worst five: -10 .. -6
    assert metrics.expected_shortfall(x) == pytest.approx(8.0)
    assert metrics.expected_shortfall(x.iloc[:20]) == pytest.approx(10.0)  # at least one month


def test_holding_less_without_timing_wins_nothing_at_matched_volatility():
    """A run at half of buy-and-hold, every month, halves the drawdown, the shortfall and the worst month.
    Matched to its volatility, buy-and-hold has the same numbers: the size alone wins nothing."""
    b = pd.Series(np.random.default_rng(2).normal(0.005, 0.045, 120))
    m = pd.DataFrame({"excess": 0.5 * b, "bench_excess": b, "size": 0.5})
    r = metrics.risk_row(m)
    assert r["max_drawdown"] > r["bh_max_drawdown"] and r["es"] < r["bh_es"]           # raw: it looks safer
    assert r["max_drawdown"] == pytest.approx(r["bh_max_drawdown_matched"])
    assert r["es"] == pytest.approx(r["bh_es_matched"])
    assert r["worst_month"] == pytest.approx(r["bh_worst_month_matched"])
    assert r["sortino"] == pytest.approx(r["bh_sortino"])                               # scale-free


def test_cost_drag_is_zero_without_costs_and_positive_at_etf_costs(setup):
    free, paid = _run(setup, "A0"), _run(setup, "A0", costs=e.COSTS["etf"])
    assert metrics.cost_drag(free, free.index)["cost_drag_sharpe"] == pytest.approx(0.0, abs=1e-12)
    d = metrics.cost_drag(paid, paid.index)
    assert d["costs_a_year"] > 0 and d["cost_drag_sharpe"] > 0


def test_the_users_calls_are_recorded_and_a2_is_gone():
    """Entry 20: SPY only, volatility stability primary (Sortino secondary at 0.16), A0 and A1."""
    assert isinstance(TRACK_A, Frozen) and isinstance(OPEN, Frozen)
    assert dict(OPEN) == {"primary_metric": "vol_stability", "books": "spy", "meaningful_gain": 0.10,
                          "formal_date": "36 months after the freeze", "a2_grade_b": "dropped"}
    require_decided()
    assert TRACK_A["family"] == list(sizing.VARIANTS) == ["A0", "A1"]
    assert OPEN["primary_metric"] in TRACK_A["metric_options"]
    with pytest.raises(ValueError, match="unknown variant"):
        sizing.resize(pd.DataFrame({"size": [1.0]}, index=pd.period_range("2001-01", periods=1, freq="M")), "A2")
    assert ev.TEST_RULES["level"] == 0.05                             # the primary test's level, reused


# ---------------------------------------------------------------- volatility stability (Entry 20)

def _garch_world(years=100, seed=0, a=0.10, b=0.89, vol=0.16):
    """A GARCH world with its true parameters: the daily panel, and each month's expected variance given the close
    before it (the sum of the GARCH forecasts of its days' variances) and its first day's variance x its days."""
    d = syn.garch(years=years, world="flat", seed=seed, a=a, b=b, vol=vol, start="1800-01-01")
    g = d.groupby(d.index.to_period("M"))
    first, n = g["sigma2"].first(), g.size()
    L, phi = vol ** 2 / 252, a + b
    expected = pd.Series([L * k + (s - L) * (1 - phi ** k) / (1 - phi) for s, k in zip(first, n)], index=first.index)
    return d, expected, first * n, L


def _true_run(d, month_var, k, L):
    """Size = c / σ^k on the given monthly variance (row t: month t + 1's), no cap, no costs, through the engine."""
    book = e.Book.single(d["x"], d["rf"], "x")
    fc = lambda p: month_var.shift(-1).dropna().rename("forecast")
    res = e.run(book, {1: "inv_vol", 2: "inv_var"}[k], forecaster=fc, c=(L * 21) ** (k / 2), max_size=None)
    periods = ex.holding_periods(book)
    return metrics.vol_stability_test(metrics.monthly_vol(book, res, periods, res.index))


@pytest.fixture(scope="module")
def world():
    return _garch_world()


def test_on_the_true_volatility_inverse_volatility_steadies_risk_and_inverse_variance_does_not(world):
    """The user's derivation (Entry 20): ln σ_timed = ln c + (1 - k) x + e. With x the true expected volatility, e is
    uncorrelated with x, so k = 1 leaves only Var(e) (R well below 1) and k = 2 leaves Var(x) + Var(e), as buy-and-hold
    does (R close to 1). Through the engine, with drift inside each month; no cap, so the test checks the measure."""
    d, expected, _, L = world
    k1, k2 = _true_run(d, expected, 1, L), _true_run(d, expected, 2, L)
    assert k1["R"] < 0.75 and k1["reading"] == "edge"                  # 0.674 here
    assert abs(k2["R"] - 1) < 0.03 and k2["reading"] != "edge"         # 1.001 here


def test_inverse_variance_on_a_forecast_that_ignores_mean_reversion_makes_risk_less_steady(world):
    """The derivation's assumption matters. The next day's variance x the month's days ignores mean reversion within
    the month, so its error falls when it rises (correlation about -0.18). Then k = 2 overshoots both ways: R above 1."""
    d, _, next_day, L = world
    assert _true_run(d, next_day, 2, L)["R"] > 1.1                      # about 1.19


def test_the_measure_is_scale_free_and_the_power_check_reads_as_entry_20_says():
    rng = np.random.default_rng(6)
    months = pd.period_range("1990-01", periods=240, freq="M")
    bh = pd.Series(np.exp(rng.normal(-3, 0.4, 240)), months)
    same = metrics.vol_stability_test(pd.DataFrame({"timed": 0.5 * bh, "bh": bh}))   # half size, no timing
    assert same["R"] == pytest.approx(1.0) and same["reduction"] == pytest.approx(0.0, abs=1e-12)
    assert same["reading"] != "edge"
    g, se, months_dev = 0.25, 0.02, 870
    expect = norm.cdf(g / (se * np.sqrt(months_dev / 36)) - 1.6448536)
    assert metrics.power_at(g, se, months_dev) == pytest.approx(expect)
    assert metrics.power_at(0.5, 0.01, 870) > 0.99 and metrics.power_at(0.0, 0.05, 870) < 0.06
