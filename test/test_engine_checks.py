"""The plan's five engine checks, on simulated data only (Phase 5: no real-data strategy result is read).

    1 two worlds     world A (constant μ) shows alpha; world B (μ proportional to σ²) shows none
    2 constant       a constant forecast gives scaled buy-and-hold: zero alpha
    3 shuffled       1,000 forecasts reordered in time earn alphas centered on zero; real timing
                     beats 95% of them in world A and does not in world B (the placebo rule)
    4 oracle         next month's actual variance: never worse than the baseline where timing pays,
                     and a large Sharpe jump where the past cannot forecast the variance
    5 hand check     five months recomputed in a spreadsheet match to floating point

Each world is one 500-year history, seed 0. 500 years is chosen for power, not for the result:
over 90 years world A's alpha reaches t > 1.96 only about half the time (Monte Carlo in
reports/phase5_engine.ipynb), so a 90-year test would fail on a correct engine.
"""

import numpy as np
import pandas as pd
import pytest

from src import engine as e
from src.engine import evaluation as ev, forecast as fc, handcheck as hc, synthetic as syn

YEARS = 500


@pytest.fixture(scope="module")
def world_a():
    d = syn.garch(years=YEARS, world="flat", seed=0)
    return e.Book.single(d["x"], d["rf"])


@pytest.fixture(scope="module")
def world_b():
    d = syn.garch(years=YEARS, world="var", seed=0)
    return e.Book.single(d["x"], d["rf"])


def test_1_alpha_in_world_a_and_none_in_world_b(world_a, world_b):
    a, b = e.run(world_a), e.run(world_b)
    sa, sb = ev.spanning(a["excess"], a["bench_excess"]), ev.spanning(b["excess"], b["bench_excess"])
    assert sa["alpha"] > 0 and sa["t_alpha"] > 1.96
    assert sb["t_alpha"] < 1.96
    assert ev.sharpe(a["excess"]) > ev.sharpe(a["bench_excess"])          # timing pays in A ...
    assert ev.sharpe(b["excess"]) < ev.sharpe(b["bench_excess"])          # ... and costs Sharpe in B


def test_2_a_constant_forecast_is_scaled_buy_and_hold(world_a):
    const = fc.constant(0.002)
    res = e.run(world_a, forecaster=const)                     # real-time c_t: size exactly 1
    assert np.allclose(res["size"], 1.0, rtol=0, atol=1e-12)
    assert np.allclose(res["excess"], res["bench_excess"], rtol=0, atol=1e-15)
    res = e.run(world_a, forecaster=const, c=0.7 * 0.002)      # fixed c: size 0.7
    assert np.allclose(res["size"], 0.7)
    assert np.allclose(res["excess"], 0.7 * res["bench_excess"], rtol=0, atol=1e-14)
    sp = ev.spanning(res["excess"], res["bench_excess"])
    assert abs(sp["alpha"]) < 1e-12 and sp["beta"] == pytest.approx(0.7, rel=1e-10) and sp["r2"] == pytest.approx(1)


def test_3_shuffled_forecasts_earn_nothing_and_real_timing_beats_them(world_a, world_b):
    draws = e.ENGINE_RULES["placebo_draws"]
    pa, pb = e.placebo(world_a, draws=draws, seed=1), e.placebo(world_b, draws=draws, seed=1)
    for p in (pa, pb):
        s = p["shuffled"]
        assert len(s) == draws
        assert abs(s.mean()) < 0.1 * s.std()                   # centered on zero, relative to its spread
    assert pa["pass"] and pa["share_beaten"] >= e.ENGINE_RULES["placebo_share"]
    assert not pb["pass"]


def test_4_the_oracle_bounds_what_a_forecast_can_add(world_a):
    base, oracle = e.run(world_a), e.run(world_a, forecaster=fc.oracle)
    sr = lambda x: ev.sharpe(x)
    assert sr(base["bench_excess"]) < sr(base["excess"]) < sr(oracle["excess"])
    d = syn.unpredictable(years=YEARS, seed=0)                 # the past says nothing about next month
    book = e.Book.single(d["x"], d["rf"])
    base, oracle = e.run(book), e.run(book, forecaster=fc.oracle)
    assert ev.spanning(base["excess"], base["bench_excess"])["t_alpha"] < 1.96
    assert sr(oracle["excess"]) - sr(oracle["bench_excess"]) >= 0.1   # a large Sharpe jump


def test_5_hand_check_matches_the_spreadsheet():
    daily, check = hc.read()
    res = hc.engine_run(daily)
    engine = np.array([res.loc[pd.Period(m, "M"), q] for m, q in zip(check["month"], check["quantity"])])
    assert len(check) == len(hc.CHECK_MONTHS) * len(hc.QUANTITIES)
    assert np.allclose(engine, check["spreadsheet"].astype(float), rtol=1e-11, atol=1e-13)
