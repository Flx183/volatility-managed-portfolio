"""
Engine stage 4: execution
=========================
Turns target positions into the returns a brokerage account would show.

The lag. The size decided at the close of month t is traded at that close (or `delay` trading days
later) and held until the next trade. Holding period m runs from the trade after month m-1 to the
trade after month m, so its return never depends on a forecast made with its own data. With
delay=0, holding period m is exactly calendar month m. A weekly book (Phase 7) does the same with
weeks ending on Friday.

Drift. Between trades nothing is rebalanced, so an instrument that gains becomes a bigger share
of the account. With w the positions put on at the trade and R, R_f the period's total returns:

    R_p = Σ w_i R_i + (1 - Σ w_i) R_f,      drifted w̃_i = w_i (1 + R_i) / (1 + R_p)

Costs.
- Trading: κ_i x |w_new,i - w̃_i| of the account per trade, paid when the new positions go on.
  The account starts in cash, so the first trade pays for the whole entry.
- Borrowing: when Σ w_i > 1 the cash leg is negative and pays rf + borrow_spread a year.
- Shorting: a short position pays short_fee a year on its value.
Both accrue daily (per trading day, 252 a year) over the holding period.

Zero-cost factors (the paper track) are treated as assets whose total return is rf + their excess
return. Before costs this gives exactly Σ w_i x_i in excess of rf at the trade; it only touches
second-order terms of the drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.engine.data import Book

DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Costs:
    kappa: dict[str, float] = field(default_factory=dict)      # one-way cost per unit of weight traded
    borrow_spread: float = 0.0                                  # a year, over rf, on borrowed cash
    short_fee: dict[str, float] = field(default_factory=dict)   # a year, on the value of a short

    def kappa_for(self, cols) -> pd.Series:
        return pd.Series(self.kappa, dtype=float).reindex(cols).fillna(0.0)


NO_COSTS = Costs()
# The cost model of the plan. ETFs: 5 bp a trade for SPY, 10 bp for factor ETFs; borrowed cash pays
# rf + 0.5% a year. The SPY short behind the MTUM tilt pays 0.25% a year in the base case: SPY is easy
# to borrow, and the plan asks only for "a small fee", so this number is a choice, not a quote. Paper factors are
# before costs (an upper bound): French and AQR factors are not tradeable as built.
# Phase 8's cost row adds "etf_short_stress": the same costs with a 1% SPY short fee, for a short whose
# rebate is low or missing.
ETF_KAPPA = {"spy": 5e-4, "mtum": 10e-4, "qual": 10e-4, "usmv": 10e-4}
COSTS = {
    "paper": NO_COSTS,
    "etf": Costs(kappa=ETF_KAPPA, borrow_spread=0.005, short_fee={"spy": 0.0025}),
    "etf_short_stress": Costs(kappa=ETF_KAPPA, borrow_spread=0.005, short_fee={"spy": 0.01}),
}


@dataclass(frozen=True)
class Periods:
    """Growth of $1 over each holding period, per instrument and for cash, under one cost model.
    The fees compound daily, so they are built in here; simulate() reads κ from the same `costs`,
    so a run has exactly one cost model."""
    costs: Costs
    growth: pd.DataFrame          # Π (1 + x + rf), long positions
    growth_short: pd.DataFrame    # Π (1 + x + rf + short_fee / 252), short positions
    cash: pd.Series               # Π (1 + rf), cash held
    cash_borrow: pd.Series        # Π (1 + rf + borrow_spread / 252), cash borrowed
    opens: pd.Series              # date of the trade that opens the period
    closes: pd.Series             # date of the trade that closes it
    days: pd.Series               # trading days in the period


def period_labels(index: pd.DatetimeIndex, delay: int = 0, freq: str = "M") -> pd.PeriodIndex:
    """The holding period of each trading day. Period m opens at the close of period m-1's last
    trading day plus `delay` trading days, and closes at the same point after period m. Periods are
    calendar months (freq "M") or weeks ending on Friday ("W-FRI")."""
    period = index.to_period(freq)
    last = pd.Series(np.arange(len(index))).groupby(period).max()         # position of each period's last day
    bounds = (last + delay).clip(upper=len(index) - 1).drop_duplicates(keep="first")
    return bounds.index[np.searchsorted(bounds.to_numpy(), np.arange(len(index)), side="left")]


def holding_periods(book: Book, costs: Costs = NO_COSTS, delay: int = 0) -> Periods:
    x, rf = book.returns, book.rf.reindex(book.returns.index)
    labels = period_labels(x.index, delay, book.freq)
    fee = pd.Series(costs.short_fee, dtype=float).reindex(x.columns).fillna(0.0) / DAYS_PER_YEAR

    def grow(daily):                                          # a period with any missing day has no growth
        out = np.exp(np.log1p(daily).groupby(labels).sum())
        return out.mask(daily.isna().groupby(labels).any())

    days = pd.Series(x.index, index=x.index)
    return Periods(
        costs=costs,
        growth=grow(x.add(rf, axis=0)),
        growth_short=grow(x.add(rf, axis=0) + fee),
        cash=grow(rf),
        cash_borrow=grow(rf + costs.borrow_spread / DAYS_PER_YEAR),
        opens=days.groupby(labels).min().shift(1).rename("opens"),       # the previous period's last day
        closes=days.groupby(labels).max().rename("closes"),
        days=days.groupby(labels).size().rename("days"),
    )


def simulate(periods: Periods, targets: pd.DataFrame) -> pd.DataFrame:
    """Returns of holding `targets` (holding months x instruments, as fractions of the account),
    under the periods' cost model.

    Columns: gross (before any cost), financing (borrow spread and short fees), pre_trade (after
    financing), turnover (Σ |trade| at the opening trade), trading (its cost), net, rf, excess
    (net - rf), and per instrument w_<i> (held) and drift_<i> (drifted, at the period's end)."""
    m = targets.index
    if len(m) and (m != pd.period_range(m[0], periods=len(m), freq=m.freq)).any():
        raise ValueError("holding periods must be consecutive")
    w, cols = targets, targets.columns
    held = w != 0
    g_long, g_short = periods.growth.loc[m, cols], periods.growth_short.loc[m, cols]
    missing = (g_long.isna() & held).any(axis=1)              # only a held instrument needs a return
    if missing.any():
        raise ValueError(f"a held instrument has no return in {missing.index[missing][0]}")
    g_long, g_short = g_long.where(held, 1.0), g_short.where(held, 1.0)
    g = g_long.where(w >= 0, g_short)                       # shorts pay the fee
    cash = 1 - w.sum(axis=1)
    g_cash = periods.cash.loc[m].where(cash >= 0, periods.cash_borrow.loc[m])

    end = (w * g).sum(axis=1) + cash * g_cash                # account growth before trading cost
    gross = (w * g_long).sum(axis=1) + cash * periods.cash.loc[m]
    drift = (w * g).div(end, axis=0)
    before = drift.shift(1).fillna(0.0)                      # the account starts in cash
    trade = (w - before).abs()
    trading = (trade * periods.costs.kappa_for(cols)).sum(axis=1)
    net = (1 - trading) * end - 1
    rf = periods.cash.loc[m] - 1
    out = pd.DataFrame({"gross": gross - 1, "financing": gross - end, "pre_trade": end - 1,
                        "turnover": trade.sum(axis=1), "trading": trading, "net": net, "rf": rf,
                        "excess": net - rf})
    return pd.concat([out, w.add_prefix("w_"), drift.add_prefix("drift_")], axis=1)
