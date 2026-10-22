"""Track B: the variants' sizes, their month-end signals and the family's tests, on simulated data only. No
Track B variant has been run on real data (PREREGISTRATION Entry 17)."""

import numpy as np
import pandas as pd
import pytest

from src import engine as e
from src.engine import execution as ex, forecast, synthetic as syn
from src.engine.data import MONTH
from src.fixes import OPEN, TRACK_B, build, require_decided, rules
from src.walkforward import runs, stats

PAIRS = 60


@pytest.fixture(scope="module")
def setup():
    d = syn.garch(years=24, world="vol", seed=3, start="1980-01-01")
    book = e.Book.single(d["x"], d["rf"], "x")
    periods = ex.holding_periods(book)
    base = runs.prepare(book, forecast.last_rv)
    R = {v: rules.run(base, periods, v, min_pairs=PAIRS) for v in rules.VARIANTS if v != "B2"}
    return book, periods, base, R


def test_the_frozen_variant_is_the_frozen_runner_to_the_last_digit(setup):
    book, periods, base, R = setup
    pd.testing.assert_frame_equal(R["frozen"], runs.run(base, 1.0, periods, min_pairs=PAIRS))


def test_b1_is_half_size_one_and_half_the_inverse_volatility_size(setup):
    """Entry 20: Wang and Yan scale by inverse volatility, so B1's managed half is c_t / σ̂ with its own c_t."""
    book, periods, base, R = setup
    own = runs.run(base, 0.5, periods, min_pairs=PAIRS)
    np.testing.assert_allclose(R["B1"]["size"], 0.5 + 0.5 * own["size"])
    np.testing.assert_allclose(R["B1"]["excess"], 0.5 * own["bench_excess"] + 0.5 * own["excess"], rtol=0, atol=1e-15)
    assert not np.allclose(own["size"], R["frozen"]["size"])          # not the frozen inverse-variance size
    assert R["B1"]["size"].max() <= 0.5 + 0.5 * TRACK_B["cap"]


def test_b2_is_b1_on_downside_volatility_with_its_own_c_t(setup):
    book, periods, _, _ = setup
    down = runs.prepare(book, runs.downside(forecast.last_rv, MONTH))
    b2 = rules.run(down, periods, "B2", min_pairs=PAIRS)
    own = runs.run(down, 0.5, periods, min_pairs=PAIRS)                # Phase 7's downside-volatility rule
    np.testing.assert_allclose(b2["size"], 0.5 + 0.5 * own["size"])
    np.testing.assert_allclose(b2["forecast"], own["forecast"])


def test_the_rank_is_the_share_of_the_last_120_forecasts_at_or_below_this_one_and_reads_no_later_month():
    months = pd.period_range("1990-01", periods=300, freq="M")
    f = pd.Series(np.random.default_rng(0).lognormal(0, 0.5, 300), index=months)
    tab = pd.DataFrame({"forecast": f, "ret": 0.0, "rf": 0.0})
    s = rules.signals(tab, min_months=60)
    assert s["rank"].iloc[:59].isna().all() and s["rank"].iloc[59:].notna().all()
    t = 200
    window = f.iloc[t - 119:t + 1]
    assert s["rank"].iloc[t] == pytest.approx((window <= f.iloc[t]).mean())
    later = tab.copy()
    later.iloc[t + 1:, 0] *= 10                                     # the future changes; month t's signals do not
    pd.testing.assert_series_equal(rules.signals(later, 60).iloc[:t + 1]["rank"], s.iloc[:t + 1]["rank"])
    full = s["extreme"].iloc[119:].astype(bool)
    assert full.mean() == pytest.approx(0.4, abs=0.05)              # 24 of 120 at each end, about 40% of months


def test_the_trend_is_the_books_value_against_the_mean_of_its_last_ten_month_ends():
    months = pd.period_range("2000-01", periods=14, freq="M")
    ret = pd.Series([0.01] * 10 + [-0.2, 0.0, 0.3, 0.0], index=months)
    s = rules.signals(pd.DataFrame({"forecast": 1.0, "ret": ret, "rf": 0.0}), min_months=1)
    level = (1 + ret).cumprod()
    assert s["above"].iloc[:9].isna().all()
    assert bool(s["above"].iloc[9]) is True and bool(s["above"].iloc[10]) is False
    assert bool(s["above"].iloc[12]) == bool(level.iloc[12] > level.iloc[3:13].mean())


def test_b3_holds_one_in_normal_months_and_b4_never_less_than_one_above_the_trend(setup):
    _, _, base, R = setup
    (seg, prep), = base
    sig = rules.decision_signals(R["frozen"], prep, PAIRS)
    extreme, above = sig["extreme"].astype(bool), sig["above"].astype(bool)
    f = R["frozen"]["size"]
    np.testing.assert_allclose(R["B3"]["size"], f.where(extreme, 1.0))
    np.testing.assert_allclose(R["B4"]["size"], np.maximum(f, 1.0).where(above, f))
    assert (R["B4"]["size"][above] >= 1).all() and R["B4"]["size"].max() <= TRACK_B["cap"]
    assert 0.2 < extreme.mean() < 0.6 and 0.3 < above.mean() < 0.95
    for v in ("B1", "B3", "B4"):
        assert R[v].index.equals(R["frozen"].index)


def test_a_screened_book_reads_each_years_recipe_and_skips_its_cash_years():
    """The paper mix: a new recipe at every year-end, and a year the screen leaves empty (cash, a new segment).
    Each month's signals come from the recipe it holds, and every variant holds the frozen run's months."""
    from src.config import FACTORS
    from src.screen import year_ends
    from src.walkforward import screened
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("1980-01-01", periods=252 * 24, name="date")
    panel = pd.DataFrame(rng.normal(0.0003, 0.01, (len(idx), len(FACTORS))), index=idx, columns=FACTORS).assign(rf=0.0001)
    years = {y: ["mkt", "bab"] for y in list(range(1990, 1996)) + list(range(1997, 2003))}   # 1996 left empty
    surv = pd.Series({d: years.get(d.year, []) for d in year_ends(panel.index)}, name="assets")
    book = screened.book(panel, screened.recipes(panel, surv), "inv_vol", MONTH)
    periods = ex.holding_periods(book)
    prep = runs.prepare(book, forecast.last_rv)
    assert len(prep) == 2                                               # two segments around the cash year
    R = {v: rules.run(prep, periods, v, min_pairs=PAIRS) for v in ("frozen", "B3", "B4")}
    assert (R["frozen"].index.year != 1997).all()                       # 1996's empty screen: 1997 in cash
    for v in ("B3", "B4"):
        assert R[v].index.equals(R["frozen"].index)
    m = pd.Period("2000-06", "M")
    seg, tabs = next((s, t) for s, t in prep if m in s.positions.index)
    k = R["frozen"].loc[m, "recipe"]
    own = rules.signals(tabs[k], PAIRS).loc[m - 1]
    expect = R["frozen"].loc[m, "size"] if own["extreme"] else 1.0
    assert R["B3"].loc[m, "size"] == pytest.approx(expect)


def test_the_family_reads_romano_wolf_and_holm_in_the_order_of_the_evidence():
    """Romano-Wolf's p comes from the bootstrap and the raw p from the normal approximation, so a weak variant's
    Romano-Wolf p can sit just below its raw p (as in Phase 7). What must hold: Holm never lowers a p, Romano-Wolf's
    p rises as the t falls, and a clear gain survives both."""
    rng = np.random.default_rng(4)
    b = rng.normal(0.005, 0.04, 240)
    shift = {"B1": 0.0, "B2": 0.0005, "B3": 0.001, "B4": 0.01}
    X = {v: pd.DataFrame({"excess": b * 0.8 + shift[v] + rng.normal(0, 0.01, 240), "bench_excess": b, "size": 0.8})
         for v in TRACK_B["family"]}
    T = build.family_tests(X, blocks=(6,))
    p = T["p"].astype(float)
    np.testing.assert_allclose(T["p_holm"], stats.holm(p.to_numpy()))
    assert (T["p_holm"] >= p - 1e-12).all()
    by_t = T.assign(t=T["gain"].astype(float) / T["se"].astype(float)).sort_values("t", ascending=False)
    assert by_t["p_rw"].is_monotonic_increasing
    assert bool(T.loc["B4", "survives_rw"]) and bool(T.loc["B4", "survives_holm"])
    assert list(T.index) == TRACK_B["family"]


def test_the_users_calls_are_recorded_and_each_variant_has_its_power():
    """Entry 20: SPY primary (development on the twin), B1 exploratory, B1 and B2 at power 0.5."""
    assert dict(OPEN) == {"primary_book": "spy", "b1_grade_b": "exploratory"}
    require_decided()
    assert build.DEV_FOR[OPEN["primary_book"]] == "twin"
    assert dict(TRACK_B["power"]) == {"frozen": 1.0, "B1": 0.5, "B2": 0.5, "B3": 1.0, "B4": 1.0}
    assert dict(TRACK_B["grade_b"]) == {"B1": "exploratory", "B2": "clean", "B3": "clean", "B4": "exploratory"}
    assert TRACK_B["family"] == ["B1", "B2", "B3", "B4"] and rules.VARIANTS[0] == "frozen"
