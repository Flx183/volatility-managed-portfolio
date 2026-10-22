"""
Track A, metrics: the primary test and the risk numbers beside it
=================================================================
    sortino_difference(a, b)        Sortino(a) - Sortino(b), a year, and its standard error: the primary Sharpe
                                    test's delta method (Ledoit and Wolf) on the moments (μ_a, μ_b, E min(a, 0)²,
                                    E min(b, 0)²), with the same Newey-West covariance (6 lags)
    sortino_gain_test(m, meaningful)  one-sided at 5%, the three readings and the minimum detectable effect, as
                                    ev.sharpe_gain_test. `meaningful` is the gain a one-sided 95% upper bound must
                                    fall below to read "no meaningful edge", on the Sortino scale (OPEN)
    expected_shortfall(x)           the mean loss of the worst 5% of months (at least one month), as a positive number
    matched(b, x)                   buy-and-hold scaled to the timed run's realized volatility over the same months
    risk_row(m)                     every risk number Track A reports for one run's scored months
    cost_drag(res, months)          what costs take from the Sharpe and Sortino gains (monthly runs)

The primary metric (Entry 20): volatility stability.
    monthly_vol(book, res, periods, months)  σ_m = √(Σ r²) over each month's trading days, for the timed account and
                                    buy-and-hold, from the daily ledger (crises.path: each day's return relative to cash)
    vol_stability_test(v)           R = SD(ln σ_m timed) / SD(ln σ_m buy-and-hold) and the reduction g = 1 - R. The p
                                    is Entry 6's paired stationary bootstrap (both columns resampled by the same months;
                                    mean blocks 6 and 12; the larger p); se is the larger bootstrap standard deviation of
                                    g; the one-sided 95% bounds are g ± 1.645 se and the MDE 2.4865 se. Readings: edge if
                                    p < 0.05; no meaningful edge if the upper bound is below 0.10; inconclusive otherwise
    power_at(g, se, months, at)     Entry 20's power check: Φ(g / (se √(months / at)) - 1.645), the chance a 36-month
                                    sample reads "edge" if the development reduction is the truth

Scale. The Sortino ratio of excess returns, like the Sharpe ratio, does not change when a run is scaled. The
other numbers do: a run held at half size has about half the drawdown, the shortfall and the worst month
whether or not its timing helps. So each is also reported for buy-and-hold at matched volatility, and the
comparison that counts is the timed run against that (TRACK_A["matched"]). Matching uses both runs'
realized volatility over the same months, so it is a measurement, not a strategy anyone could hold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.engine import evaluation as ev
from src.engine.data import MONTH
from src.overlay import TRACK_A
from src.robustness import crises
from src.walkforward import stats

VOL_BOUND = 0.10                         # no meaningful edge: the reduction's one-sided 95% upper bound below this (Entry 20)


def newey_west(y: np.ndarray, lags: int) -> np.ndarray:
    """The Newey-West (Bartlett) long-run covariance of the columns of y, each already centered."""
    n = len(y)
    psi = y.T @ y / n
    for j in range(1, lags + 1):
        g = y[j:].T @ y[:-j] / n
        psi += (1 - j / (lags + 1)) * (g + g.T)
    return psi


def sortino_difference(a: pd.Series, b: pd.Series, lags: int = ev.TEST_RULES["lags"]) -> tuple[float, float]:
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    n = len(x)
    d = np.minimum(x, 0) ** 2
    mu, lpm = x.mean(axis=0), d.mean(axis=0)
    grad = np.array([1 / np.sqrt(lpm[0]), -1 / np.sqrt(lpm[1]), -mu[0] / (2 * lpm[0] ** 1.5), mu[1] / (2 * lpm[1] ** 1.5)])
    y = np.column_stack([x - mu, d - lpm])
    se = np.sqrt(grad @ newey_west(y, lags) @ grad / n) * np.sqrt(12)
    return float((mu[0] / np.sqrt(lpm[0]) - mu[1] / np.sqrt(lpm[1])) * np.sqrt(12)), float(se)


def sortino_gain_test(m: pd.DataFrame, meaningful: float, lags: int = ev.TEST_RULES["lags"]) -> dict:
    """The Sortino gain of the timed run over buy-and-hold, from monthly excess returns (excess, bench_excess)."""
    gain, se = sortino_difference(m["excess"], m["bench_excess"], lags)
    z_level, z_power = norm.ppf(1 - ev.TEST_RULES["level"]), norm.ppf(ev.TEST_RULES["power"])
    out = {"gain": gain, "se": se, "z": gain / se, "p": norm.sf(gain / se), "lower": gain - z_level * se,
           "upper": gain + z_level * se, "mde": (z_level + z_power) * se,
           "months": int(m[["excess", "bench_excess"]].dropna().shape[0])}
    if out["p"] < ev.TEST_RULES["level"]:
        reading = "edge"
    elif out["upper"] < meaningful:
        reading = "no meaningful edge"
    else:
        reading = "inconclusive"
    return {**out, "reading": reading}


def expected_shortfall(x: pd.Series, level: float = TRACK_A["es_level"]) -> float:
    k = max(1, int(np.ceil(level * len(x))))
    return float(-np.sort(x.to_numpy())[:k].mean())


def matched(b: pd.Series, x: pd.Series) -> pd.Series:
    return b * (x.std() / b.std())


def risk_row(m: pd.DataFrame, gamma: float = TRACK_A["gamma"], level: float = TRACK_A["es_level"]) -> dict:
    """m: monthly excess, bench_excess and size on the scored months. bh_*_matched is buy-and-hold at the timed
    run's realized volatility."""
    x, b = m["excess"], m["bench_excess"]
    bm = matched(b, x)
    return {"months": len(m), "mean_size": m["size"].mean(),
            "volatility": x.std() * np.sqrt(12), "bh_volatility": b.std() * np.sqrt(12),
            "sortino": stats.sortino(x), "bh_sortino": stats.sortino(b),
            "sharpe": ev.sharpe(x), "bh_sharpe": ev.sharpe(b),
            "es": expected_shortfall(x, level), "bh_es": expected_shortfall(b, level),
            "bh_es_matched": expected_shortfall(bm, level),
            "max_drawdown": ev.max_drawdown(x), "bh_max_drawdown": ev.max_drawdown(b),
            "bh_max_drawdown_matched": ev.max_drawdown(bm),
            "worst_month": x.min(), "bh_worst_month": b.min(), "bh_worst_month_matched": bm.min(),
            "cer": ev.cer(x, gamma), "bh_cer": ev.cer(b, gamma)}


def cost_drag(res: pd.DataFrame, months: pd.PeriodIndex) -> dict:
    """Each gain before every cost (trading and financing) minus after it, on these months of a monthly run,
    and the costs a year of both runs."""
    if res.index.freqstr != MONTH:
        raise ValueError("cost_drag reads a monthly run's own months")
    r = res.loc[months]
    gross = pd.DataFrame({"excess": r["gross"] - r["rf"], "bench_excess": r["bench_gross"] - r["rf"]})
    net = r[["excess", "bench_excess"]]
    gain = lambda f, d: f(d["excess"]) - f(d["bench_excess"])
    return {"cost_drag_sharpe": gain(ev.sharpe, gross) - gain(ev.sharpe, net),
            "cost_drag_sortino": gain(stats.sortino, gross) - gain(stats.sortino, net),
            "costs_a_year": (r["trading"] + r["financing"]).mean() * 12,
            "bh_costs_a_year": (r["bench_trading"] + r["bench_financing"]).mean() * 12}


# ---------------------------------------------------------------- volatility stability (Entry 20)

def monthly_vol(book, res: pd.DataFrame, periods, months: pd.PeriodIndex) -> pd.DataFrame:
    """σ_m of the timed account and of buy-and-hold on each of `months`, from the run's daily ledger."""
    p = crises.path(book, res, periods)
    g = p.index.to_period(MONTH)
    out = pd.DataFrame({"timed": np.sqrt((p["ret"] ** 2).groupby(g).sum()),
                        "bh": np.sqrt((p["bench_ret"] ** 2).groupby(g).sum())}).reindex(months)
    if out.isna().any().any() or (out <= 0).any().any():
        raise ValueError("every scored month needs a positive realized volatility in both runs")
    return out


def _log_sd_ratio(C: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """SD(a) / SD(b) in each resample (rows of C: how often each month is drawn)."""
    n = len(a)
    var = lambda x: (C @ x ** 2 / n - (C @ x / n) ** 2) * n / (n - 1)
    return np.sqrt(var(a) / var(b))


def vol_stability_test(v: pd.DataFrame, blocks=stats.BOOT["blocks"], reps: int = stats.BOOT["reps"],
                       seed: int = stats.BOOT["seed"], bound: float = VOL_BOUND) -> dict:
    a, b = np.log(v["timed"].to_numpy()), np.log(v["bh"].to_numpy())
    R = float(np.std(a, ddof=1) / np.std(b, ddof=1))
    g = 1 - R
    ps, ses = [], []
    for blk in blocks:
        draws = 1 - _log_sd_ratio(stats.counts(len(a), blk, reps, seed), a, b)
        ps.append(float(stats.bootstrap_p(np.array([g]), draws[:, None])[0]))
        ses.append(float(draws.std(ddof=1)))
    p, se = max(ps), max(ses)
    z_level, z_power = norm.ppf(1 - ev.TEST_RULES["level"]), norm.ppf(ev.TEST_RULES["power"])
    out = {"months": len(a), "R": R, "reduction": g, "se": se, "p": p, "lower": g - z_level * se,
           "upper": g + z_level * se, "mde": (z_level + z_power) * se,
           "sd_log_timed": float(np.std(a, ddof=1)), "sd_log_bh": float(np.std(b, ddof=1))}
    out["reading"] = "edge" if p < ev.TEST_RULES["level"] else ("no meaningful edge" if out["upper"] < bound else "inconclusive")
    return out


def power_at(g: float, se: float, months: int, at: int = 36) -> float:
    z_level = norm.ppf(1 - ev.TEST_RULES["level"])
    return float(norm.cdf(g / (se * np.sqrt(months / at)) - z_level))
