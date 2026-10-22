"""Track C, C1: μ̂, the size c_t x μ̂ / σ̂² through the engine, and the first sized month, on simulated data only."""

import numpy as np
import pandas as pd
import pytest

from src import engine as e
from src.engine import execution as ex, forecast, synthetic as syn
from src.engine.data import MONTH
from src.frozen import Frozen
from src.vrp import C1, TRACK_C, eda, strategy
from src.walkforward import runs


def _xy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    m = pd.period_range("1990-01", periods=n, freq=MONTH)
    x = pd.Series(rng.normal(0.001, 0.001, n), m)
    y = pd.Series(np.r_[2.0 * x.to_numpy()[:-1] + rng.normal(0.005, 0.04, n - 1), np.nan], m)   # y_t: month t + 1
    return x, y


def test_mu_hat_is_the_gates_out_of_sample_forecast_shrunk_then_floored_and_needs_no_later_return():
    x, y = _xy()
    M = strategy.mu_hat(x, y)
    P = eda.oos_predictions(x, y, 1, TRACK_C["oos_min_pairs"])
    common = P.index
    np.testing.assert_allclose(M.loc[common, "forecast"], P["pred"])        # the gate's evidence covers this rule
    np.testing.assert_allclose(M.loc[common, "mean"], P["bench"])
    assert M.index[0] == pd.Period("1995-01", MONTH) and M.index[-1] == x.index[-1]   # the last month needs no y
    np.testing.assert_allclose(M["mu"], np.maximum(0.0, 0.5 * M["forecast"] + 0.5 * M["mean"]))
    assert C1["mu_min_pairs"] == TRACK_C["oos_min_pairs"] == 60
    t = pd.Period("2000-06", MONTH)
    later = y.copy()
    later.loc[t:] += 1.0                                       # y_t is month t + 1's return: not known at t's close
    pd.testing.assert_frame_equal(strategy.mu_hat(x, later).loc[:t], M.loc[:t])


def test_the_floor_applies_after_shrinking():
    m = pd.period_range("1990-01", periods=80, freq=MONTH)
    x = pd.Series(np.r_[np.linspace(-1, 1, 79), -5.0], m)     # a last VRP far below the rest
    y = pd.Series(np.r_[0.01 + 0.01 * x.to_numpy()[:-1], np.nan], m)
    row = strategy.mu_hat(x, y).iloc[-1]
    assert row["forecast"] < 0 and row["mean"] > 0
    assert row["mu"] == pytest.approx(max(0.0, 0.5 * row["forecast"] + 0.5 * row["mean"]))


@pytest.fixture(scope="module")
def sim():
    d = syn.garch(years=12, world="vol", seed=4, start="1990-01-01")
    book = e.Book.single(d["x"], d["rf"], "x")
    months = pd.period_range("1990-01", "2001-12", freq=MONTH)
    mu = pd.Series(np.abs(np.random.default_rng(5).normal(0.006, 0.004, len(months))), months)
    mu.loc[:"1994-12"] = np.nan                                 # μ̂ starts in January 1995, as with 60 pairs from 1990
    mu.loc["1999-03"] = 0.0                                     # a floored month
    periods = ex.holding_periods(book)
    res = runs.run(runs.prepare(book, strategy.forecaster(mu, forecast.last_rv)), 1.0, periods,
                   max_size=C1["cap"], min_pairs=C1["min_pairs"])
    return book, mu, res


def test_the_engine_sizes_c_t_times_mu_over_sigma_squared_capped_and_zero_when_mu_is_zero(sim):
    book, mu, res = sim
    rv = forecast.last_rv(book.returns["x"])
    decided = res.index - 1                                     # each size is set at the close before its month
    expect = (res["c"].to_numpy() * mu.reindex(decided).to_numpy() / rv.reindex(decided).to_numpy())
    np.testing.assert_allclose(res["size"], np.minimum(expect, C1["cap"]))
    assert res.loc["1999-04", "size"] == 0.0                     # μ̂ of 0 at March's close: no position in April
    assert (res["size"] >= 0).all() and res["size"].max() <= C1["cap"]


def test_the_first_sized_month_is_february_1998_when_mu_starts_in_january_1995(sim):
    """36 pairs (signal at s, return of s + 1) from January 1995: the 36th ends with January 1998's return, so c_t
    first exists at January 1998's close and the first sized month is February 1998."""
    _, _, res = sim
    assert res.index[0] == pd.Period("1998-02", MONTH)


def test_the_c1_rules_are_read_only_and_match_the_users_calls():
    assert isinstance(C1, Frozen)
    assert (C1["mu_returns"], C1["mu_model"], C1["mu_horizon"], C1["min_pairs"], C1["cap"]) == ("mkt", "M4", 1, 36, 1.5)
    assert (C1["shrink"], C1["floor"]) == (0.5, 0.0)
