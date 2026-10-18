"""
M0-M3: models that update a variance from past squared returns
==============================================================
    M0 rolling   Σ r² over the last 22 trading days (the papers' baseline)
    M1 ewma      σ̂²_{d+1} = λ σ̂²_d + (1 - λ) r²_d, λ = 0.94; the forecast is flat, so 22 x σ̂²
    M2 garch     σ²_d = ω + a ε²_{d-1} + b σ²_{d-1}; h-step forecasts summed over h = 1..22
    M3 gjr       GARCH plus g ε²_{d-1} 1{ε_{d-1} < 0}: an extra kick after down days

GARCH estimation: Gaussian quasi-maximum likelihood on the daily returns in percent, with a zero
mean (so ε = r, the same r² as RV and EWMA), refit at every month-end on all data up to it. The
recursion starts at the mean r² of that data. Constraints: ω > 0, a, b >= 0, a + g >= 0 (a down day
never lowers variance) and persistence ρ = a + b + g/2 < 1, so shocks fade. The h-step forecast
decays toward the long-run level σ̄² = ω / (1 - ρ):

    E_t[σ²_{t+h}] = σ̄² + ρ^(h-1) (σ²_{t+1} - σ̄²),   summed:  H σ̄² + (σ²_{t+1} - σ̄²)(1 - ρ^H)/(1 - ρ)

For GJR, ρ uses g/2 because a shock is negative half the time when shocks are symmetric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter

from src.models.inputs import H, at_origins, schedule

LAMBDA = 0.94          # RiskMetrics
SCALE = 100.0          # GARCH is fitted on returns in percent, for better-conditioned parameters
MAX_PERSISTENCE = 0.9999


def rolling(p: pd.Series) -> pd.Series:
    p = p.dropna()
    return at_origins((p ** 2).rolling(H).sum(), p)


def ewma(p: pd.Series, lam: float = LAMBDA) -> pd.Series:
    """The recursion started at the first r², written as a filter. Dividing by 1 - λ^(d+1) makes the
    weights sum to one from the first day (it matters only in the first years, long before a forecast)."""
    p = p.dropna()
    r2 = (p ** 2).to_numpy()
    s2_next = lfilter([1 - lam], [1, -lam], r2) / (1 - lam ** np.arange(1, len(r2) + 1))
    return at_origins(H * s2_next, p)


# ---------------------------------------------------------------- GARCH

def garch_path(theta: np.ndarray, r2: np.ndarray, neg: np.ndarray, init: float) -> np.ndarray:
    """σ²_d for every day, each from data up to d-1. theta = (ω, a, g, b); neg = r² on down days, else 0."""
    w, a, g, b = theta
    u = np.empty_like(r2)
    u[0] = init
    u[1:] = w + a * r2[:-1] + g * neg[:-1]
    return lfilter([1.0], [1.0, -b], u)


def _nll(theta, r2, neg, init):
    s2 = garch_path(theta, r2, neg, init)
    if not np.all(s2 > 0):
        return 1e12
    return 0.5 * np.sum(np.log(s2) + r2 / s2)


def garch_fit(r2: np.ndarray, neg: np.ndarray, init: float, gjr: bool,
              start: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """Quasi-maximum likelihood (SLSQP). Returns (ω, a, g, b), with g = 0 for plain GARCH, and success.

    Two safeguards, added on 2026-10-18 after the first real-data run (PREREGISTRATION Entry 5): ω is
    optimized as a share of the mean r², so every parameter is of order 0.01 to 1 (SLSQP is sensitive to
    scale), with ω at most the mean r²; and the result is kept only if its likelihood beats the start's.
    In the first run SLSQP reported success at points far worse than its start (ω 10⁹ times too large)."""
    v = r2.mean()
    scale = np.array([v, 1.0, 1.0, 1.0])
    if gjr:
        pack, unpack = (lambda t: t), (lambda x: x)
        bounds = [(1e-6, 1), (0, 1), (-1, 1), (0, 1)]
    else:
        pack, unpack = (lambda t: t[[0, 1, 3]]), (lambda x: np.array([x[0], x[1], 0.0, x[2]]))
        bounds = [(1e-6, 1), (0, 1), (0, 1)]
    if start is None:
        start = np.array([0.02 * v, 0.08, 0.04 if gjr else 0.0, 0.90])
    start = np.array(start, float)                            # a copy: the caller's estimate is not touched
    start[0] = min(start[0], v)
    nll = lambda theta: _nll(theta, r2, neg, init)
    cons = [{"type": "ineq", "fun": lambda x: MAX_PERSISTENCE - persistence(unpack(x))},
            {"type": "ineq", "fun": lambda x: unpack(x)[1] + unpack(x)[2]}]
    res = minimize(lambda x: nll(unpack(x) * scale), pack(start / scale), method="SLSQP",
                   bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
    theta = unpack(res.x) * scale
    if not nll(theta) <= nll(start):
        return start, False
    return theta, bool(res.success)


def persistence(theta: np.ndarray) -> float:
    w, a, g, b = theta
    return a + b + g / 2


def garch_sum(theta: np.ndarray, s2_next: float, h: int = H) -> float:
    """Σ_{k=1..h} E_t[σ²_{t+k}], from the one-step forecast σ²_{t+1}."""
    rho = persistence(theta)
    long_run = theta[0] / (1 - rho)
    return h * long_run + (s2_next - long_run) * (1 - rho ** h) / (1 - rho)


def garch(p: pd.Series, gjr: bool = False, every: str = "month") -> pd.Series:
    """M2 (gjr=False) or M3 (gjr=True): the forecast only (the engine's forecaster)."""
    return garch_fits(p, gjr, every)[0]


def garch_fits(p: pd.Series, gjr: bool = False, every: str = "month") -> tuple[pd.Series, pd.DataFrame]:
    """The forecasts and the parameters of each refit. Each refit starts from the last estimate and
    never ends at a worse likelihood than it. `converged` is False where the optimizer stopped without
    meeting its tolerance; that estimate is still at least as good as the last one."""
    p = p.dropna()
    r = p.to_numpy() * SCALE
    r2 = r ** 2
    neg = np.where(r < 0, r2, 0.0)
    o, refit = schedule(p, every)
    out, params, theta = [], [], None
    for month, pos, re in zip(o.index, o.to_numpy(), refit):
        n = pos + 1                                           # data up to the origin's close
        init = r2[:n].mean()
        if re or theta is None:
            theta, ok = garch_fit(r2[:n], neg[:n], init, gjr, theta)
            params.append({"month": month, "omega": theta[0], "a": theta[1], "g": theta[2], "b": theta[3], "converged": ok})
        s2 = garch_path(theta, r2[:n], neg[:n], init)[-1]     # σ² of the origin day
        w, a, g, b = theta
        s2_next = w + a * r2[n - 1] + g * neg[n - 1] + b * s2
        out.append(garch_sum(theta, s2_next) / SCALE ** 2)
    return pd.Series(out, index=o.index, name="forecast"), pd.DataFrame(params).set_index("month")
