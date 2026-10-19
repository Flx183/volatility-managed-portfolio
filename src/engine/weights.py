"""
Engine stage 3: weights
=======================
    signal(forecast, power)        1/σ̂² (power 1) or 1/σ̂ (power 0.5)
    realtime_c(ret, signal)        c_t = sd(r_{s+1}) / sd(r_{s+1} x signal_s), over pairs with s+1 <= t
    target_c(annual_vol)           c for a fixed volatility target (Barroso and Santa-Clara: 12%)
    size(c, signal, max_size)      c_t x signal_t, kept between 0 and max_size

Index convention: a value in row t is known at the close of month t. The size in row t is held
during month t+1; the engine makes that shift, never this module.

Why c_t in real time: Moreira and Muir set c on the full sample, which uses future data. With a
fixed c and no cap the Sharpe ratio ignores c, but c_t drifts and meets the cap, so it changes
results. Real-time estimation is where Cederburg et al. (2020) find the gains leak away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine import data


def signal(forecast: pd.Series, power: float) -> pd.Series:
    f = forecast.dropna()
    if (f <= 0).any():
        raise ValueError(f"variance forecasts must be positive; first bad month {f.index[f <= 0][0]}")
    return (f ** -power).rename("signal")


def realtime_c(ret: pd.Series, sig: pd.Series, min_months: int = 120, window: int | None = None) -> pd.Series:
    """c_t that matches the managed and unmanaged volatility over the months up to t.

    Pair s holds the signal known at the end of month s and the return of month s+1. Pair s is
    complete at the end of month s+1, so c_t uses pairs s <= t-1. None until min_months pairs exist.
    window=None uses every pair so far; an integer uses the last `window` pairs. For a daily book,
    "month" reads "trading day", and s+1 is the next trading day of ret's calendar."""
    nxt = ret.copy()
    nxt.index = data.shift(ret.index, -1, ret.index)           # row s: r_{s+1}
    nxt = nxt[nxt.index.notna()]
    pairs = pd.DataFrame({"y": nxt, "z": nxt * sig}).dropna()
    roll = pairs.expanding(min_months) if window is None else pairs.rolling(window, min_periods=min_months)
    sd = roll.std()
    c = (sd["y"] / sd["z"]).dropna()
    c.index = data.shift(c.index, 1, ret.index)                # pair s is complete at the end of s+1
    c = c[c.index.notna()]
    return c.reindex(c.index.union(sig.index)).ffill().reindex(sig.index).rename("c")


def target_c(annual_vol: float) -> float:
    """c for signal 1/σ̂ (power 0.5) with σ̂² a monthly variance: size = (annual_vol / √12) / σ̂."""
    return annual_vol / np.sqrt(12)


def size(c: pd.Series | float, sig: pd.Series, max_size: float | None) -> pd.Series:
    """c_t x signal_t, floored at 0 and capped at max_size (None: no cap)."""
    w = (c * sig).dropna()
    return w.clip(lower=0, upper=max_size).rename("size")
