"""
Simulated worlds for the engine tests
=====================================
Daily excess returns x_d = μ_d + σ_d z_d, with z_d standard normal and σ²_d a GARCH(1,1):

    σ²_d = ω + a ε²_{d-1} + b σ²_{d-1},     ε_d = σ_d z_d

The worlds share the same shocks z (so the same variance path) and differ only in how the
expected return moves with risk. Each has the same average expected return.

    "flat"  μ_d constant              the best weight μ/(γσ²) falls with σ²: inverse variance wins
    "vol"   μ_d proportional to σ_d   the best weight falls with σ: inverse volatility wins
    "var"   μ_d proportional to σ²_d  the best weight is constant: timing cannot help

The plan's world A is "flat" and world B is "var". The GARCH parameters (a = 0.10, b = 0.89) put the
monthly RV of a simulated market near the real one (Phase 2): standard deviation of log RV 0.83
(market 1.04), AR(1) of RV 0.72 (0.62), wildest month 44 times the median (52).

unpredictable() is a fourth world, for the oracle test: each month draws its volatility
independently, so last month's variance says nothing about next month's. There the baseline can
earn nothing, and only a forecast that sees the future (the oracle) can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WORLDS = ("flat", "vol", "var")


def garch(years: float = 90, world: str = "flat", premium: float = 0.06, vol: float = 0.16, a: float = 0.10,
          b: float = 0.89, rf: float = 0.02, seed: int = 0, start: str = "1700-01-01", burn: int = 1000) -> pd.DataFrame:
    """Daily frame: x (excess return), rf, sigma2 (true conditional variance), mu (true expected return).
    premium and vol are annual: the average expected excess return and the long-run volatility.
    The calendar is business days from `start`; pandas dates end in 2262, so at most ~560 years."""
    if world not in WORLDS:
        raise ValueError(f"world must be one of {WORLDS}")
    n = int(round(years * 252))
    z = np.random.default_rng(seed).standard_normal(n + burn)
    long_run = vol ** 2 / 252
    omega = long_run * (1 - a - b)
    s2 = np.empty(n + burn)
    s2[0] = long_run
    for d in range(1, n + burn):
        s2[d] = omega + a * s2[d - 1] * z[d - 1] ** 2 + b * s2[d - 1]
    s2, z = s2[burn:], z[burn:]
    risk = {"flat": np.ones(n), "vol": np.sqrt(s2), "var": s2}[world]
    mu = premium / 252 * risk / risk.mean()                     # same average expected return in every world
    idx = _business_days(start, n, years)
    return pd.DataFrame({"x": mu + np.sqrt(s2) * z, "rf": rf / 252, "sigma2": s2, "mu": mu}, index=idx)


def unpredictable(years: float = 90, premium: float = 0.06, vol: float = 0.16, vol_of_vol: float = 0.5,
                  rf: float = 0.02, seed: int = 0, start: str = "1700-01-01") -> pd.DataFrame:
    """Constant μ; each calendar month's volatility drawn independently, lognormal with standard
    deviation vol_of_vol of log volatility, scaled so the long-run volatility is `vol`.
    Same columns as garch()."""
    n = int(round(years * 252))
    idx = _business_days(start, n, years)
    month = idx.to_period("M")
    codes, months = pd.factorize(month)
    rng = np.random.default_rng(seed)
    log_vol = np.log(vol) - vol_of_vol ** 2 + vol_of_vol * rng.standard_normal(len(months))   # E[σ²] = vol²
    s2 = np.exp(2 * log_vol)[codes] / 252
    mu = np.full(n, premium / 252)
    return pd.DataFrame({"x": mu + np.sqrt(s2) * rng.standard_normal(n), "rf": rf / 252, "sigma2": s2, "mu": mu},
                        index=idx)


def _business_days(start: str, n: int, years: float) -> pd.DatetimeIndex:
    days = pd.date_range(start, "2262-04-01", freq="D", name="date")     # pandas' last representable dates
    idx = days[days.dayofweek < 5][:n]
    if len(idx) < n:
        raise ValueError(f"{years} years of business days from {start} run past pandas' last date (2262)")
    return idx
