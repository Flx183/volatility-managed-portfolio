"""
Engine stage 2: forecast
========================
A forecaster takes the daily excess returns of the timed mix and returns one number per month t:
the variance forecast for month t+1, made at the close of month t's last trading day from data up
to that close. Units are monthly (a month's sum of squared daily returns).

    last_rv        RV_t, last month's realized variance (Moreira and Muir; the baseline)
    last_rv_down   RV⁻_t, from down days only (Wang and Yan; Phase 2's downside RV)
    rv_days(126)   21 x mean r² over the last 126 trading days (Barroso and Santa-Clara)
    constant(x)    the same number every month (engine test: zero alpha)
    oracle         RV_{t+1}, next month's actual variance. Looks ahead on purpose (engine test)

A new model is any function with the same signature, passed to the engine as `forecaster`;
nothing downstream changes. permute() reorders a forecast in time (the shuffled-forecast placebo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import variance as v


def last_rv(p: pd.Series) -> pd.Series:
    return v.monthly_table(p)["rv"].rename("forecast")


def last_rv_down(p: pd.Series) -> pd.Series:
    return v.monthly_table(p)["rv_down"].rename("forecast")


def rv_days(days: int = 126):
    """Barroso and Santa-Clara: monthly variance from the last `days` daily returns, 21 x mean r²."""
    def forecaster(p: pd.Series) -> pd.Series:
        p = p.dropna()
        mean_r2 = (p ** 2).rolling(days).mean()
        out = 21 * mean_r2.groupby(p.index.to_period("M")).last()
        return out.loc[v.monthly_table(p).index].rename("forecast")
    forecaster.__name__ = f"rv_{days}d"
    return forecaster


def constant(value: float):
    def forecaster(p: pd.Series) -> pd.Series:
        return pd.Series(value, index=v.monthly_table(p).index, name="forecast")
    forecaster.__name__ = f"constant_{value:g}"
    return forecaster


def oracle(p: pd.Series) -> pd.Series:
    """Next month's realized variance, known in advance: a cheat, as an upper bound."""
    return v.monthly_table(p)["rv"].shift(-1).rename("forecast")


oracle.looks_ahead = True


def permute(forecast: pd.Series, seed: int) -> pd.Series:
    """The same forecast values, randomly reordered in time."""
    f = forecast.dropna()
    return pd.Series(np.random.default_rng(seed).permutation(f.to_numpy()), index=f.index, name=f.name)
