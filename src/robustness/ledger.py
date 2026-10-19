"""
Phase 8, the daily ledger
=========================
Every Phase 8 run, whatever its rebalancing, is booked day by day and scored on calendar months
(PREREGISTRATION Entry 8: "scored on calendar months"; the fix of Entry 12):

    days(book, res, periods)       every trading day of the run's holding periods: its period, and each
                                   account's value relative to the account at the period's opening trade,
                                   after financing and before the trading cost (simulate's drift, day by day)
    pieces(book, res, periods)     the days of one holding period that fall in one calendar month: each
                                   account's growth over them, the cash's, and whether they open the period
    monthly(pieces, res, kappa)    calendar-month excess returns of both accounts, with each period's trading
                                   cost paid on its first day: the run's own cost, or κ x turnover; rf; and
                                   the size held, averaged over the month's days
    months(book, res, periods)     monthly(pieces(...)), with whether the run holds every trading day of each month
    whole(book, res)               the calendar months the run holds on every trading day

A monthly period is its own calendar month, so its months are the engine's to the last digit. A weekly or
daily period that straddles a month-end is split at it, and each day's return goes to its own month. So
"the month is held on every trading day" and "the month's return" cover the same days at every frequency.
(Phase 7's runs.scored gives each week to the month of its closing trade; Phase 7's rows keep that rule.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import engine as e
from src.engine import execution
from src.engine.data import MONTH


def days(book: e.Book, res: pd.DataFrame, periods: execution.Periods) -> pd.DataFrame:
    """Columns period, cash (rf growth since the opening trade), value and bench_value. On a period's last day
    value is simulate's 1 + pre_trade, bit for bit."""
    x, rf = book.returns, book.rf.reindex(book.returns.index)
    labels = pd.Series(execution.period_labels(x.index, 0, book.freq), index=x.index)
    labels = labels[labels.isin(res.index)]
    cols = [c[2:] for c in res.columns if c.startswith("w_")]
    X, R = x.loc[labels.index, cols], rf.loc[labels.index]
    fee = pd.Series(periods.costs.short_fee, dtype=float).reindex(cols).fillna(0.0) / execution.DAYS_PER_YEAR
    borrow = periods.costs.borrow_spread / execution.DAYS_PER_YEAR
    g = labels.to_numpy()
    grow = lambda daily: np.exp(np.log1p(daily).groupby(g).cumsum())          # as holding_periods, day by day
    out = {"period": labels, "cash": grow(R)}
    XR = X.add(R, axis=0)
    long, short = grow(XR), grow(XR + fee)
    for pre in ("", "bench_"):
        W = res.loc[g, [f"{pre}w_{c}" for c in cols]].set_axis(cols, axis=1).set_axis(labels.index)
        held = long.where(W >= 0, short).where(W != 0, 1.0)                  # shorts pay the fee
        left = 1 - W.sum(axis=1)
        out[f"{pre}value"] = (W * held).sum(axis=1) + left * out["cash"].where(left >= 0, grow(R + borrow))
    return pd.DataFrame(out)


def pieces(book: e.Book, res: pd.DataFrame, periods: execution.Periods) -> pd.DataFrame:
    """One row per (holding period, calendar month) in time order: period, month, opens (the piece holds the
    period's first day), days, and the growth over the piece of value, bench_value and cash."""
    d = days(book, res, periods)
    d["month"] = d.index.to_period(MONTH)
    by = d.groupby(["period", "month"], sort=False)
    last = by.last()
    opens = ~last.index.get_level_values("period").duplicated()
    out = pd.DataFrame({"period": last.index.get_level_values("period"), "month": last.index.get_level_values("month"),
                        "opens": opens, "days": by.size().to_numpy()})
    for col in ("value", "bench_value", "cash"):
        v = last[col].to_numpy()
        out[col] = v / np.where(opens, 1.0, np.roll(v, 1))                    # a period's later piece: since its last one
    return out


def monthly(P: pd.DataFrame, res: pd.DataFrame, kappa: float | None = None) -> pd.DataFrame:
    """Columns excess, bench_excess, rf and size, by calendar month. kappa replaces the run's own trading cost
    with κ per unit traded (for the break-even)."""
    month = P["month"].to_numpy()
    cash = P["cash"].groupby(month).prod()
    out = {}
    for pre in ("", "bench_"):
        cost = res[f"{pre}trading"] if kappa is None else kappa * res[f"{pre}turnover"]
        c = np.where(P["opens"], cost.reindex(P["period"]).to_numpy(), 0.0)    # paid at the opening trade
        growth = pd.Series((1 - c) * P[f"{pre}value"].to_numpy()).groupby(month).prod()
        out[f"{pre}excess"] = (growth - 1) - (cash - 1)
    size = res["size"].reindex(P["period"]).to_numpy() * P["days"].to_numpy()
    out["rf"] = cash - 1
    out["size"] = pd.Series(size).groupby(month).sum() / P["days"].groupby(month).sum()
    m = pd.DataFrame(out)
    m.index = pd.PeriodIndex(m.index, freq=MONTH)
    return m


def whole(book: e.Book, res: pd.DataFrame) -> pd.PeriodIndex:
    """The calendar months whose every trading day the run holds a position. A monthly run holds whole months;
    a weekly or daily run can start or stop inside one."""
    labels = pd.Series(execution.period_labels(book.returns.index, 0, book.freq), index=book.returns.index)
    held = labels.isin(res.index)
    w = held.groupby(book.returns.index.to_period(MONTH)).all()
    return pd.PeriodIndex(w.index[w.to_numpy()], freq=MONTH)


def months(book: e.Book, res: pd.DataFrame, periods: execution.Periods, kappa: float | None = None) -> pd.DataFrame:
    m = monthly(pieces(book, res, periods), res, kappa)
    return m.assign(whole=m.index.isin(whole(book, res)))
