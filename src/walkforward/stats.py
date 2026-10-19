"""
Phase 7, step 3: the tests
==========================
    counts(n, block, reps, seed)       the stationary bootstrap (Politis and Romano 1994), as how many
                                       times each month is drawn in each resample
    sharpe_draws, cer_draws            Sharpe ratio and CER of every column in every resample
    sortino(x)                         mean / downside deviation, a year
    bootstrap_p(stat, draws)           one-sided p of a gain: share of centered draws at or above it
    combination(managed, bench)        Cederburg et al.'s real-time three-way split and its benchmark
    romano_wolf(t, t_draws)            stepdown adjusted p-values (one-sided)
    holm(p)                            Holm-Bonferroni adjusted p-values
    deflated_sharpe(x, ...)            Bailey and López de Prado (2014), measured against buy-and-hold

Why a block bootstrap. Resampling single months independently would destroy the volatility
clustering the strategy exploits. The stationary bootstrap resamples blocks of random length (mean 6
or 12 months here), so calm and wild stretches stay together. All variants are resampled with the
same draws, which keeps their correlation: what Romano-Wolf needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

BOOT = {"blocks": (6, 12), "reps": 10_000, "seed": 0}      # PREREGISTRATION Entry 6
COMBINATION = {"gamma": 5, "train": 120, "max_abs": 5.0}    # Cederburg et al. (2020), as in the plan
EULER = 0.5772156649015329


# ---------------------------------------------------------------- the bootstrap

def counts(n: int, block: float, reps: int = BOOT["reps"], seed: int = BOOT["seed"]) -> np.ndarray:
    """reps x n: how often each of n months appears in each stationary-bootstrap resample. A resample
    starts at a random month and walks forward, jumping to a fresh random month with probability
    1/block at each step (so blocks have geometric lengths with mean `block`), wrapping at the end."""
    rng = np.random.default_rng(seed)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    jump = rng.random((reps, n)) < 1 / block
    fresh = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return np.bincount((np.arange(reps)[:, None] * n + idx).ravel(), minlength=reps * n).reshape(reps, n)


def _moments(C: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance (ddof 1) of every column of X in every resample."""
    n = X.shape[0]
    m1, m2 = C @ X / n, C @ (X ** 2) / n
    return m1, (m2 - m1 ** 2) * n / (n - 1)


def sharpe_draws(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    m1, var = _moments(C, X)
    return m1 / np.sqrt(var) * np.sqrt(12)


def cer_draws(C: np.ndarray, X: np.ndarray, gamma: float) -> np.ndarray:
    m1, var = _moments(C, X)
    return 12 * (m1 - gamma / 2 * var)


def bootstrap_p(stat: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """One-sided p-value of H0 "gain <= 0": the share of resampled gains, centered on the real one,
    that reach the real gain."""
    return ((draws - stat) >= stat).mean(axis=0)


# ---------------------------------------------------------------- metrics and tests

def sortino(x: pd.Series) -> float:
    """Mean excess return over the downside deviation √mean(min(x, 0)²), a year."""
    return x.mean() / np.sqrt((np.minimum(x, 0) ** 2).mean()) * np.sqrt(12)


def combination(managed: pd.Series, bench: pd.Series, gamma: float = COMBINATION["gamma"],
                train: int = COMBINATION["train"], max_abs: float = COMBINATION["max_abs"]) -> pd.DataFrame:
    """Cederburg et al.'s real-time test. Each month t after the first `train`, a mean-variance investor
    with risk aversion γ splits the account among buy-and-hold, the managed run and cash, with weights
    Σ̂⁻¹μ̂ / γ from the months before t (expanding). If the absolute weights sum to more than max_abs,
    they are scaled down to max_abs. The benchmark investor holds only buy-and-hold and cash:
    μ̂ / (γ σ̂²), clipped to ±max_abs. Returns both monthly excess returns and the weights."""
    X = pd.concat([bench.rename("bench"), managed.rename("managed")], axis=1).dropna()
    A = X.to_numpy()
    rows = []
    for t in range(train, len(A)):
        past = A[:t]
        mu, S = past.mean(axis=0), np.cov(past, rowvar=False)
        w = np.linalg.solve(S, mu) / gamma
        if np.abs(w).sum() > max_abs:
            w *= max_abs / np.abs(w).sum()
        w_alone = np.clip(mu[0] / (gamma * S[0, 0]), -max_abs, max_abs)
        rows.append({"combined": w @ A[t], "alone": w_alone * A[t, 0], "w_bench": w[0], "w_managed": w[1],
                     "w_alone": w_alone})
    return pd.DataFrame(rows, index=X.index[train:])


def romano_wolf(t: np.ndarray, t_draws: np.ndarray) -> np.ndarray:
    """Romano and Wolf's stepdown adjusted p-values for one-sided tests (large t is evidence).
    t: the K studentized statistics. t_draws: reps x K bootstrap statistics centered on the real ones,
    (θ* - θ̂) / se. Step j compares the j-th largest t with the largest bootstrap statistic among the
    hypotheses not yet rejected; p-values are made non-decreasing down the order. The chance of any
    false rejection stays at the level, whatever the correlation between the tests."""
    order = np.argsort(-t)
    tail_max = np.maximum.accumulate(t_draws[:, order][:, ::-1], axis=1)[:, ::-1]   # max over steps j..K
    p = np.maximum.accumulate((tail_max >= t[order]).mean(axis=0))
    out = np.empty(len(t))
    out[order] = p
    return out


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values: sorted p_(i) times (m - i + 1), made non-decreasing, capped at
    1. A hypothesis is rejected at 5% when its adjusted p is below 0.05, which is the stepwise rule
    p_(i) <= 0.05 / (m - i + 1), stopping at the first failure."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out


def expected_max(trials: int) -> float:
    """E[max of `trials` independent standard normals], Bailey and López de Prado's approximation."""
    return (1 - EULER) * norm.ppf(1 - 1 / trials) + EULER * norm.ppf(1 - 1 / (trials * np.e))


def deflated_sharpe(x: pd.Series, bench_sharpe: float, gain_var: float, trials: int) -> dict:
    """Bailey and López de Prado's deflated Sharpe ratio of the selected variant, measured against its
    buy-and-hold. Monthly units throughout (not annualized).

    threshold = buy-and-hold's Sharpe ratio + √(variance of the trials' Sharpe gains) x E[max of N normals]:
    the Sharpe ratio the best of N useless variants would reach by luck. Then
        DSR = Φ( (SR - threshold) √(T - 1) / √(1 - skew SR + (kurtosis - 1)/4 SR²) ),
    the probability that the true Sharpe ratio is above the threshold, allowing for fat tails.
    DSR >= 0.95 is the usual bar."""
    sr = x.mean() / x.std()
    g3, g4 = skew(x), kurtosis(x, fisher=False)
    threshold = bench_sharpe + np.sqrt(gain_var) * expected_max(trials)
    z = (sr - threshold) * np.sqrt(len(x) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"sharpe_monthly": sr, "bh_sharpe_monthly": bench_sharpe, "threshold_monthly": threshold,
            "expected_max": expected_max(trials), "trials": trials, "skew": g3, "kurtosis": g4,
            "months": len(x), "dsr": float(norm.cdf(z))}
