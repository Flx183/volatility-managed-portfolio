"""
Engine stage 5: evaluation
==========================
Everything here reads the engine's monthly output and nothing else.

    sharpe(excess)               annualized Sharpe ratio of monthly excess returns
    sharpe_gain_test(res)        the primary test: Sharpe gain over buy-and-hold, one-sided at 5%,
                                 its 95% bounds, the three readings and the minimum detectable effect
    spanning(managed, bench)     managed = α + β bench + ε: α a year, White t, β, appraisal ratio
    summary(res)                 timed vs buy-and-hold: mean, realized volatility, Sharpe, drawdown,
                                 turnover, costs, and the timed run's average size
    break_even(res)              the one-way κ at which the timed Sharpe ratio falls to buy-and-hold's
    zero_profit_cost(...)        the one-way κ at which a strategy's mean excess return is zero
    placebo_share(real, draws)   share of shuffled-forecast alphas the real alpha beats

Both kinds of break-even are reported (a round trip is 2κ). Neither is compared with Barroso and
Santa-Clara's 1.27 and 1.75 points: those are the costs that make momentum's profits insignificant
at 5%, and they appear to include the trading inside the factor, which the engine does not price.

Realized volatility is reported for both runs beside every Sharpe ratio, so a gap in a
volatility-dependent measure (such as a certainty-equivalent return) cannot hide a gap in size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from src import variance as v

# The primary test (PREREGISTRATION Entry 3), fixed before any real-data result under it.
TEST_RULES = {
    "level": 0.05,             # one-sided: H0 is "timing does not raise the Sharpe ratio"
    "meaningful_gain": 0.10,   # Sharpe a year: an upper 95% bound below this reads "no meaningful edge"
    "power": 0.80,             # the minimum detectable effect is reported at this power
    "lags": v.NW_MONTHLY,      # Newey-West lags of the HAC covariance (monthly returns)
}


def sharpe(excess: pd.Series) -> float:
    return excess.mean() / excess.std() * np.sqrt(12)


def sharpe_gain_test(res: pd.DataFrame, lags: int = TEST_RULES["lags"]) -> dict:
    """The primary test of timing: gain = Sharpe(timed) - Sharpe(buy-and-hold), a year, from the
    run's monthly excess returns (after the run's costs).

    Standard error: the delta method of Ledoit and Wolf (2008) on the moments (μ1, μ2, E r1², E r2²),
    with a Newey-West covariance, so fat tails and autocorrelation do not shrink it. The test is
    one-sided at 5%. Readings, in this order:
        "edge"                the test rejects (the 95% lower bound is above 0)
        "no meaningful edge"  the one-sided 95% upper bound is below 0.10 a year
        "inconclusive"        neither
    mde: the gain the test would detect with 80% probability, (z_0.95 + z_0.80) x standard error.
    Realized volatility of both runs and the timed run's average size are reported beside it."""
    x = res[["excess", "bench_excess"]].dropna().to_numpy()
    n = len(x)
    mu, m2 = x.mean(axis=0), (x ** 2).mean(axis=0)
    var = m2 - mu ** 2
    grad = np.array([m2[0] / var[0] ** 1.5, -m2[1] / var[1] ** 1.5, -mu[0] / (2 * var[0] ** 1.5), mu[1] / (2 * var[1] ** 1.5)])
    y = np.column_stack([x - mu, x ** 2 - m2])
    psi = y.T @ y / n
    for j in range(1, lags + 1):
        gamma = y[j:].T @ y[:-j] / n
        psi += (1 - j / (lags + 1)) * (gamma + gamma.T)
    se = np.sqrt(grad @ psi @ grad / n) * np.sqrt(12)
    gain = sharpe(res["excess"]) - sharpe(res["bench_excess"])
    z_level, z_power = norm.ppf(1 - TEST_RULES["level"]), norm.ppf(TEST_RULES["power"])
    out = {"gain": gain, "se": se, "z": gain / se, "p": norm.sf(gain / se), "lower": gain - z_level * se,
           "upper": gain + z_level * se, "mde": (z_level + z_power) * se, "months": n,
           "volatility: timed": res["excess"].std() * np.sqrt(12),
           "volatility: buy-and-hold": res["bench_excess"].std() * np.sqrt(12), "mean size": res["size"].mean()}
    return {**out, "reading": reading(out)}


def reading(test: dict) -> str:
    if test["p"] < TEST_RULES["level"]:
        return "edge"
    if test["upper"] < TEST_RULES["meaningful_gain"]:
        return "no meaningful edge"
    return "inconclusive"


def max_drawdown(excess: pd.Series) -> float:
    wealth = (1 + excess).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def spanning(managed: pd.Series, bench: pd.Series) -> dict:
    res = v.fit(managed.rename("managed"), bench.rename("bench"), "white")
    resid_sd = np.sqrt(res.mse_resid)
    return {"alpha": res.params["const"] * 12, "t_alpha": res.tvalues["const"], "beta": res.params["bench"],
            "r2": res.rsquared, "appraisal": res.params["const"] / resid_sd * np.sqrt(12), "months": int(res.nobs)}


def alpha(managed: pd.Series, bench: pd.Series) -> float:
    """The spanning α a year, without standard errors (fast: for the placebo's 1,000 draws)."""
    y, x = managed.to_numpy(), bench.to_numpy()
    beta = np.cov(y, x)[0, 1] / x.var(ddof=1)
    return (y.mean() - beta * x.mean()) * 12


def summary(res: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, pre in (("timed", ""), ("buy-and-hold", "bench_")):
        ex = res[f"{pre}excess"]
        rows[name] = {"mean": ex.mean() * 12, "volatility": ex.std() * np.sqrt(12), "sharpe": sharpe(ex),
                      "max_drawdown": max_drawdown(ex), "turnover_a_year": res[f"{pre}turnover"].iloc[1:].mean() * 12,
                      "costs_a_year": (res[f"{pre}trading"] + res[f"{pre}financing"]).mean() * 12}
    rows["timed"].update({"mean_size": res["size"].mean(), "share_capped": res["capped"].mean()})
    return pd.DataFrame(rows).T


def excess_at(res: pd.DataFrame, kappa: float, prefix: str = "") -> pd.Series:
    """Monthly excess returns if every trade cost κ per unit traded (replacing the run's own κ)."""
    return (1 - kappa * res[f"{prefix}turnover"]) * (1 + res[f"{prefix}pre_trade"]) - 1 - res["rf"]


def break_even(res: pd.DataFrame, hi: float = 0.5) -> float:
    """One-way κ at which the timed Sharpe ratio equals buy-and-hold's. 0 if timing does not beat
    buy-and-hold before trading costs; inf if the gain survives κ = hi."""
    gap = lambda k: sharpe(excess_at(res, k)) - sharpe(excess_at(res, k, "bench_"))
    if gap(0.0) <= 0:
        return 0.0
    if gap(hi) > 0:
        return np.inf
    return brentq(gap, 0.0, hi, xtol=1e-10)


def zero_profit_cost(res: pd.DataFrame, prefix: str = "", hi: float = 0.5) -> float:
    """One-way κ at which the strategy's mean excess return is zero (Barroso and Santa-Clara's kind)."""
    mean_at = lambda k: excess_at(res, k, prefix).mean()
    if mean_at(0.0) <= 0:
        return 0.0
    if mean_at(hi) > 0:
        return np.inf
    return brentq(mean_at, 0.0, hi, xtol=1e-12)


def placebo_share(real: float, shuffled: np.ndarray) -> float:
    """Share of the shuffled alphas that the real alpha beats. The placebo rule passes at 0.95."""
    return float((np.asarray(shuffled) < real).mean())
