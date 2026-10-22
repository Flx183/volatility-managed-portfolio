"""
Track C, C1: equity timing on SPY with the variance risk premium
================================================================
The size is Moreira and Muir's w = μ / σ², with VRP supplying μ (Entries 20 and 21):

    size_t = c_t x μ̂_t / σ̂²_t, capped at 1.5

    mu_hat(x, y)                 at each month-end t: the regression of next month's excess return on VRP, fitted on
                                 the pairs complete by t's close (s + 1 <= t) once 60 exist, its forecast at VRP_t, the
                                 training mean, and μ̂_t = max(0, 0.5 x forecast + 0.5 x mean). Needs no return after t
    twin_mu(panel)               μ̂ from the twin: the market factor's M4 VRP (as in the EDA) and its monthly excess
                                 returns, from January 1990. First μ̂: January 1995
    forecaster(mu, variance)     the rule as an engine forecaster. The engine sizes c_t x forecast⁻¹, so C1 hands it
                                 σ̂²_t / μ̂_t (infinite when μ̂_t = 0, a size of 0): the size is then c_t x μ̂_t / σ̂²_t,
                                 with the engine's real-time c_t, cap, costs and execution unchanged
    specs(panel)                 the books C2 runs: C1 on SPY (scored), C1 on the twin and the volatility-only SPY
                                 rule (beside)

Every input at month t is known at t's close: VIX, the M4 and M2 forecasts, and the pairs that end by then.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import engine as e
from src import models as md
from src.engine import data
from src.models.scorecard import SPY_COSTS
from src.vrp import C1, eda
from src.walkforward import tracks


def mu_hat(x: pd.Series, y: pd.Series, h: int = C1["mu_horizon"], min_pairs: int = C1["mu_min_pairs"],
           shrink: float = C1["shrink"], floor: float = C1["floor"]) -> pd.DataFrame:
    """x: VRP by month. y: the excess return over the h months after each month (eda.ahead). One row per month t
    with x_t and min_pairs complete pairs before it: forecast, mean and mu. The pairs and the mean are exactly those
    of the gate's out-of-sample R² (eda.oos_predictions), but no row needs y_t."""
    pairs = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    px, py, idx = pairs["x"].to_numpy(), pairs["y"].to_numpy(), pairs.index
    rows = {}
    for t, xt in x.dropna().items():
        train = (idx + h) <= t
        if train.sum() < min_pairs:
            continue
        b, a = np.polyfit(px[train], py[train], 1)
        forecast, mean = a + b * xt, py[train].mean()
        rows[t] = {"forecast": forecast, "mean": mean, "mu": max(floor, (1 - shrink) * forecast + shrink * mean)}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["forecast", "mean", "mu"]).rename_axis("month")


def twin_mu(panel: pd.DataFrame) -> pd.DataFrame:
    """μ̂ from the twin: the market factor's M4 VRP (January 1990 on) and its next-month excess returns."""
    model = C1["mu_model"]
    vrp = eda.vrp(panel, {model: md.forecasters(panel, False)[model](panel[C1["mu_returns"]])})[f"vrp_{model}"]
    mon = data.monthly(panel[C1["mu_returns"]], panel["rf"])
    return mu_hat(vrp, eda.ahead(mon, C1["mu_horizon"])).join(vrp.rename("vrp"))


def forecaster(mu: pd.Series, variance):
    """The engine forecaster for C1: σ̂²_t / μ̂_t on the months both exist (NaN before), infinite where μ̂_t is 0."""
    def f(p: pd.Series) -> pd.Series:
        s2 = variance(p)
        m = mu.reindex(s2.index)
        with np.errstate(divide="ignore"):
            out = s2 / m
        return out.where(m.notna()).rename("forecast")
    f.__name__ = f"c1_{getattr(variance, '__name__', 'variance')}"
    return f


def specs(panel: pd.DataFrame, mu: pd.Series) -> dict[str, dict]:
    """book, forecaster, costs, c_t minimum and power for each run of C2."""
    spy = e.Book.single(panel["spy"], panel["rf"], "spy")
    twin = e.Book.single(panel["mkt"], panel["rf"], "mkt")
    spy_m2 = tracks.etf_garch(False)                       # SPY's frozen M2, from SPY's 12th month
    twin_m2 = md.forecasters(panel, False)["M2"]           # the twin's own M2 (Phase 6's SPY reference)
    n = C1["min_pairs"]
    return {
        "c1_spy": {"book": spy, "forecaster": forecaster(mu, spy_m2), "costs": e.COSTS["etf"], "min_pairs": n},
        "c1_twin": {"book": twin, "forecaster": forecaster(mu, twin_m2), "costs": SPY_COSTS, "min_pairs": n},
        "vol_spy": {"book": spy, "forecaster": spy_m2, "costs": e.COSTS["etf"], "min_pairs": n},
    }
