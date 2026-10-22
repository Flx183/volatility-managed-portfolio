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
from src.walkforward import stats


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
