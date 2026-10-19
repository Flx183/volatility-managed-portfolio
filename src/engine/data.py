"""
Engine stage 1: data
====================
What the engine trades, as a Book:

    returns    daily excess returns, one column per traded instrument
    rf         daily risk-free return
    positions  position in each instrument per unit of size, one row per holding month
               (the Phase 4 recipe; a single asset is one column of ones)

The timed object is the mix p = Σ pos_i x_i. Its forecast and its c_t are computed on the mix that
will be held next month, applied to all past data: "the variance of what you will hold". So when
the recipe changes at a year-end, January's size uses December's returns under the new recipe,
as in Phase 4.

Monthly returns are exact excess returns, compounded total return minus compounded risk-free
return, so they match what execution books. RV and downside RV come from Phase 2's monthly_table,
so every phase measures variance the same way.

Holding periods are calendar months (positions indexed by monthly periods) or, from Phase 7, weeks
ending on Friday (weekly periods): the book's freq. A weekly book trades at each week's last
trading day, and its per-period table is the same as the monthly one, over weeks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import variance as v

MONTH, WEEK = "M", "W-FRI"       # the two kinds of holding period


@dataclass(frozen=True)
class Book:
    returns: pd.DataFrame        # daily excess returns, one column per instrument
    rf: pd.Series                # daily risk-free return, same index
    positions: pd.DataFrame      # PeriodIndex of holding periods x instruments: position per unit of size

    def __post_init__(self):
        missing = set(self.positions.columns) - set(self.returns.columns)
        if missing:
            raise ValueError(f"positions name instruments with no returns: {sorted(missing)}")
        if not isinstance(self.positions.index, pd.PeriodIndex) or self.positions.index.freqstr not in (MONTH, WEEK):
            raise ValueError("positions must be indexed by monthly periods or weekly periods (W-FRI): "
                             "the period the position is held")
        if self.positions.isna().any().any():
            raise ValueError("positions contain NaN")
        if self.rf.reindex(self.returns.index).isna().any():
            raise ValueError("the risk-free rate is missing on some trading days")

    @property
    def freq(self) -> str:
        """"M" (calendar months) or "W-FRI" (weeks ending on Friday)."""
        return self.positions.index.freqstr

    @classmethod
    def single(cls, x: pd.Series, rf: pd.Series, name: str = "asset", freq: str = MONTH) -> "Book":
        """One instrument, held at one unit of size in every period of the data."""
        periods = x.dropna().index.to_period(freq).unique()
        return cls(x.to_frame(name), rf, pd.DataFrame(1.0, index=periods, columns=[name]))


def recipe_ids(positions: pd.DataFrame) -> pd.Series:
    """One id per distinct row of positions: months holding the same mix share an id."""
    return positions.groupby(list(positions.columns), sort=False).ngroup().rename("recipe")


def mix(book: Book, pos: pd.Series) -> pd.Series:
    """Daily excess return of the mix with these positions, on every day all its instruments exist."""
    pos = pos[pos != 0]
    return (book.returns[pos.index].dropna() @ pos).rename("p")


def monthly(p: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """Calendar-month table of a daily excess return: ret (compounded total minus compounded rf),
    rf, rv and rv_down. Months with fewer than 10 trading days are dropped (Phase 2's rule)."""
    p = p.dropna()
    rf = rf.reindex(p.index)
    tab = v.monthly_table(p)
    g = p.index.to_period("M")
    total = np.exp(np.log1p(p + rf).groupby(g).sum())
    cash = np.exp(np.log1p(rf).groupby(g).sum())
    out = pd.DataFrame({"ret": total - cash, "rf": cash - 1, "rv": tab["rv"], "rv_down": tab["rv_down"]})
    return out.loc[tab.index]


def weekly(p: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """The monthly table's columns over weeks ending on Friday, keeping every week with a trading day.
    rv_down is Σ r² on the week's down days (no extension to the week before)."""
    p = p.dropna()
    rf = rf.reindex(p.index)
    g = p.index.to_period(WEEK)
    total = np.exp(np.log1p(p + rf).groupby(g).sum())
    cash = np.exp(np.log1p(rf).groupby(g).sum())
    r2 = p ** 2
    return pd.DataFrame({"ret": total - cash, "rf": cash - 1, "rv": r2.groupby(g).sum(),
                         "rv_down": r2.where(p < 0, 0.0).groupby(g).sum()})


def table(p: pd.Series, rf: pd.Series, freq: str = MONTH) -> pd.DataFrame:
    """The per-period table of a daily excess return: monthly() or weekly()."""
    return monthly(p, rf) if freq == MONTH else weekly(p, rf)


def from_phase4(panel: pd.DataFrame, rec: pd.DataFrame, method: str) -> Book:
    """Book for one Phase 4 mix: the recipe set at each year-end is held in every month of the next
    calendar year, with positions pos_<instrument> per unit of serving size."""
    pos = rec.xs(method, level="method").filter(like="pos_").astype(float)
    pos.columns = pos.columns.str.removeprefix("pos_")
    months = panel.index.to_period("M").unique()
    rows = []
    for asof, row in pos.iterrows():
        held = months[months.year == asof.year + 1]
        if held.empty:                                      # the last recipe is held after the sample
            continue
        rows.append(pd.DataFrame([row.to_numpy()] * len(held), index=held, columns=pos.columns))
    return Book(panel[pos.columns], panel["rf"], pd.concat(rows))
