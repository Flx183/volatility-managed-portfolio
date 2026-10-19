"""
Phase 8, the crisis case studies
================================
Averages hide how the strategy behaves in the months that matter most. A strategy that cuts risk
after the crash and misses the recovery can look fine on average and still be painful to hold.

    EPISODES                     the four development crises (2020 and 2022 sit in the holdout: Phase 9)
    span(panel, first, last)     the market's peak (its highest close in the first year) and trough (its
                                 lowest close after the peak, by the end of the last year), and the window
                                 from 6 months before the peak to 12 months after the trough
    path(book, res, periods)     day by day through a run: the timed and buy-and-hold accounts' value
                                 relative to cash, their daily returns, and the size and forecast held
    covers(p, s, calendar)       whether a run holds a position on every day of the window
    questions(p, s)              the first two questions in numbers (below)
    managed(panel, factor, how)  one factor timed alone: Barroso and Santa-Clara's rule, or the frozen one

The three questions, fixed before any crisis was looked at (PREREGISTRATION Entry 8):
1. Did it cut risk before or after the worst days? The worst days are buy-and-hold's five lowest daily
   returns between the peak and the trough. Compare the size held on them with the mean size over the
   six months before the peak, and the timed run's loss on them with buy-and-hold's.
2. Did it miss the rebound? Growth over the 12 months after the trough, timed against buy-and-hold,
   beside the same over the fall (peak to trough), and the mean size in each.
3. How much did the cap and costs matter? The window's growth with the base cap, with no cap, at a cap of
   1.0, and at 30 bp a trade (monthly runs; the window in whole calendar months).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import engine as e
from src import models as md
from src.engine import execution, forecast, weights
from src.robustness import ROBUST_RULES
from src.walkforward import runs

# Episode -> (the year the market's peak is searched in, the last year its trough is searched in).
# The plan's "2008 to 2009" crisis began at the market's October 2007 peak, so its peak is searched in 2007.
EPISODES = {"1929-1932": ("1929", "1932"), "October 1987": ("1987", "1987"),
            "2000-2002": ("2000", "2002"), "2008-2009": ("2007", "2009")}
# Momentum's two worst stretches in Barroso and Santa-Clara (2015), for their decile spread WML. French's
# momentum factor (UMD) is a different portfolio, so its losses are not the same numbers.
MOMENTUM_CRASHES = {"July-August 1932": ("1932-07", "1932-08", -0.9159), "March-May 2009": ("2009-03", "2009-05", -0.7342)}
BSC = {"days": 126, "target": e.ENGINE_RULES["target_vol"]}     # 126-day variance, 12% a year, no cap


def span(panel: pd.DataFrame, first: str, last: str) -> dict:
    """The market's peak and trough by its total-return level (market excess return plus rf), and the
    window around them."""
    level = (1 + panel["mkt"] + panel["rf"]).loc[f"{first}-01-01":f"{last}-12-31"].cumprod()
    peak = level.loc[first].idxmax()
    trough = level.loc[peak:].idxmin()
    start = peak - pd.DateOffset(months=ROBUST_RULES["lead_months"])
    end = trough + pd.DateOffset(months=ROBUST_RULES["after_months"])
    return {"peak": peak, "trough": trough, "start": start, "end": end, "market_fall": level[trough] / level[peak] - 1,
            "months": pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")}


def path(book: e.Book, res: pd.DataFrame, periods: execution.Periods) -> pd.DataFrame:
    """Every trading day of the run's holding periods: value and bench_value (the timed and buy-and-hold
    accounts relative to cash, so 1.10 means 10% ahead of cash since the first day), ret and bench_ret
    (their daily returns relative to cash), and the size and forecast set at the period's opening trade.
    Inside a period the positions drift as execution.simulate's do, and each period's last day compounds
    to the engine's 1 + net."""
    x, rf = book.returns, book.rf.reindex(book.returns.index)
    labels = pd.Series(execution.period_labels(x.index, 0, book.freq), index=x.index)
    labels = labels[labels.isin(res.index)]
    cols = [c[2:] for c in res.columns if c.startswith("w_")]
    X, R = x.loc[labels.index, cols], rf.loc[labels.index]
    fee = pd.Series(periods.costs.short_fee, dtype=float).reindex(cols).fillna(0.0) / execution.DAYS_PER_YEAR
    borrow = periods.costs.borrow_spread / execution.DAYS_PER_YEAR
    g = labels.to_numpy()
    grow = lambda daily: np.exp(np.log(daily).groupby(g).cumsum())          # growth since the period's opening trade
    cash = grow(1 + R)
    out = {}
    for pre in ("", "bench_"):
        W = res.loc[g, [f"{pre}w_{c}" for c in cols]].set_axis(cols, axis=1).set_axis(labels.index)
        held = grow(X.add(R, axis=0) + 1).where(W >= 0, grow(X.add(R, axis=0) + 1 + fee))
        left = 1 - W.sum(axis=1)
        cash_held = cash.where(left >= 0, grow(1 + R + borrow))
        inside = (W * held).sum(axis=1) + left * cash_held
        opening = (1 + res[f"{pre}net"]).cumprod().shift(1, fill_value=1.0)    # account value at each opening trade
        value = opening.loc[g].to_numpy() * (1 - res.loc[g, f"{pre}trading"].to_numpy()) * inside
        rel = value / np.cumprod(1 + R)
        out[f"{pre}value"], out[f"{pre}ret"] = rel, rel / rel.shift(1, fill_value=1.0) - 1
    out["size"] = res.loc[g, "size"].to_numpy()
    out["forecast"] = res.loc[g, "forecast"].to_numpy() if "forecast" in res else np.nan
    return pd.DataFrame(out, index=labels.index)


def covers(p: pd.DataFrame, s: dict, calendar: pd.DatetimeIndex) -> bool:
    """Whether the run holds a position on every trading day of the episode's window."""
    days = calendar[(calendar >= s["start"]) & (calendar <= s["end"])]
    return bool(days.isin(p.index).all())


def questions(p: pd.DataFrame, s: dict, worst_days: int = ROBUST_RULES["worst_days"]) -> dict:
    """The first two questions in numbers, from a run's daily path over the whole window (see covers)."""
    lead = p.loc[s["start"]:s["peak"]].iloc[:-1]
    fall = p.loc[s["peak"]:s["trough"]]
    rebound = p.loc[s["trough"]:s["end"]]
    worst = fall["bench_ret"].nsmallest(worst_days).index
    growth = lambda v: v.iloc[-1] / v.iloc[0] - 1
    timed_worst, bench_worst = p.loc[worst, "ret"].sum(), p.loc[worst, "bench_ret"].sum()
    return {"lead_size": lead["size"].mean(), "peak_size": p.loc[s["peak"], "size"],
            "worst_size": p.loc[worst, "size"].mean(), "first_worst_day": worst.min().date(),
            "worst_days": ", ".join(str(d.date()) for d in sorted(worst)),
            "worst_timed": timed_worst, "worst_bench": bench_worst, "worst_share": timed_worst / bench_worst,
            "fall_timed": growth(fall["value"]), "fall_bench": growth(fall["bench_value"]), "fall_size": fall["size"].mean(),
            "rebound_timed": growth(rebound["value"]), "rebound_bench": growth(rebound["bench_value"]),
            "rebound_size": rebound["size"].mean()}


def window_growth(monthly_excess: pd.Series, months: pd.PeriodIndex) -> float:
    """Π(1 + excess) - 1 over the window's months (NaN if the run does not hold every one of them)."""
    x = monthly_excess.reindex(months)
    return float(np.prod(1 + x) - 1) if x.notna().all() else np.nan


def managed(panel: pd.DataFrame, factor: str, how: str, max_size: float | None | str = "rule",
            costs: e.Costs = e.NO_COSTS) -> tuple[e.Book, pd.DataFrame, execution.Periods]:
    """One factor timed alone, monthly. how="bsc": Barroso and Santa-Clara's rule, size = (12% / √12) / σ̂
    with σ̂² = 21 x the mean r² of the last 126 days, no cap, as published: it needs six months of data,
    so it is the only rule with a position in 1929-1932. how="frozen": the frozen specification (M3,
    inverse variance, real-time c_t from 120 pairs, cap 1.5). max_size="rule" keeps the rule's own cap."""
    b = e.Book.single(panel[factor], panel["rf"], factor)
    periods = execution.holding_periods(b, costs)
    if how == "bsc":
        cap = None if max_size == "rule" else max_size
        res = e.run(b, "inv_vol", forecaster=forecast.rv_days(BSC["days"]), c=weights.target_c(BSC["target"]),
                    max_size=cap, costs=costs)
        return b, res.assign(entry=res.index == res.index[0]), periods
    cap = e.ENGINE_RULES["max_size"] if max_size == "rule" else max_size
    return b, runs.run(runs.prepare(b, md.frozen(True)), 1.0, periods, max_size=cap), periods
