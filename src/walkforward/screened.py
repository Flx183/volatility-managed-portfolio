"""
Phase 7, step 1: the real-time screen and the books it gives
============================================================
At every year-end the Phase 3 screen runs on all seven factors with data up to that date. The factors
that pass filters 1, 3 and 6 (level φ, 120 months of history) are held from the first rebalance after
that year-end to the first rebalance after the next one. Filters 4 and 5 judge an ETF, and none exists
before 1993 (SPY) or 2011-2013 (the factor ETFs), so the paper track applies the factor filters only.
Its results are an upper bound, as in Phase 4 (PREREGISTRATION Entry 6).

    screen_all(panel, ohlc)          Phase 3's screen at every year-end, stacked
    survivors(table)                 the factors held after each year-end
    recipes(panel, survivors)        Phase 4 steps 1-2 on each list: duplicates, then the four mixes
    book(panel, rec, method, freq)   positions per holding month or week (no row in a year the screen
                                     left empty: the account is in cash and the period is not scored)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import engine as e
from src import portfolio as pf
from src import screen as sc
from src import variance as v
from src.config import FACTORS
from src.engine.data import MONTH

FACTOR_FILTERS = ["f1", "f3", "f6"]       # predictable risk, exploitable link, stable


def screen_all(panel: pd.DataFrame, ohlc: pd.DataFrame, jobs: int = 1) -> pd.DataFrame:
    """screen(panel, ohlc, asof) at every year-end, one block of rows per year-end (index asof, factor).
    Each call reads only data up to its year-end."""
    dates = sc.year_ends(panel.index)
    tables = Parallel(n_jobs=jobs)(delayed(sc.screen)(panel, ohlc, d) for d in dates)
    return pd.concat(tables, keys=dates, names=["asof", "factor"])


def survivors(table: pd.DataFrame) -> pd.Series:
    """For each year-end, the factors passing the factor filters with 120 months of history, in FACTORS order."""
    ok = table[FACTOR_FILTERS].all(axis=1) & (table["months"] >= sc.SCREEN_RULES["min_months"])
    dates = table.index.get_level_values("asof").unique()
    return pd.Series({d: [f for f in FACTORS if ok.loc[(d, f)]] for d in dates}, name="assets")


def recipes(panel: pd.DataFrame, assets: pd.Series) -> pd.DataFrame:
    """Phase 4's Steps 1 and 2 at each year-end whose list is not empty, from data up to that date:
    the duplicate check on all history so far (ρ > +0.7 on all or crisis months, judged on 120 common
    months), then the four mix weights from the last 10 years of daily returns. On the paper track a
    position is its mix weight. Index (asof, method); columns w_<factor> for all seven (0 if not held),
    the assets held, those dropped as duplicates, and the window start."""
    mkt_rv = v.monthly_table(panel["mkt"])["rv"]
    rows = []
    for asof, names in assets.items():
        if not names:
            continue
        hist = panel[names].loc[:asof].dropna()
        win = hist.loc[hist.index > asof - pd.DateOffset(months=pf.PORTFOLIO_RULES["window_months"])]
        history = {a: int(panel[a].loc[:asof].count()) for a in names}
        strength = lambda a: v.t4_spanning(v.monthly_table(hist[a]))["t_alpha"]
        dropped = pf.dedupe(pf.monthly_returns(hist), mkt_rv, history, strength)
        keep = [a for a in names if a not in dropped]
        for method in pf.METHODS:
            w = pf.mix_weights(win[keep], method).reindex(FACTORS, fill_value=0.0)
            rows.append({"asof": asof, "method": method, "assets": " ".join(keep),
                         "dropped": "; ".join(f"{a} ({why})" for a, why in dropped.items()),
                         "window_start": win.index[0], **w.add_prefix("w_")})
    return pd.DataFrame(rows).set_index(["asof", "method"])


def opening_trades(index: pd.DatetimeIndex, freq: str) -> pd.Series:
    """For each holding period (month, or week ending on Friday) of the trading calendar: the day of
    the trade that opens it, the last trading day of the period before."""
    days = pd.Series(index, index=index)
    closes = days.groupby(index.to_period(freq)).max()
    return closes.shift(1).dropna()


def book(panel: pd.DataFrame, rec: pd.DataFrame, method: str, freq: str = MONTH) -> e.Book:
    """Positions per holding period: the recipe set at the latest year-end on or before the trade that
    opens the period. Every year-end is a screen date, so a period after a year-end with no survivors
    has no row. A monthly book holds the recipe over the next calendar year; a weekly one switches at
    the first weekly rebalance on or after the year-end."""
    w = rec.xs(method, level="method").filter(regex="^w_").astype(float)
    w.columns = w.columns.str.removeprefix("w_")
    w = w.loc[:, (w != 0).any()]
    starts = sc.year_ends(panel.index)
    opens = opening_trades(panel.index, freq)
    k = np.searchsorted(starts.to_numpy(), opens.to_numpy(), side="right") - 1
    regime = pd.Series(starts[np.maximum(k, 0)], index=opens.index).where(k >= 0)
    regime = regime[regime.isin(w.index)]
    positions = pd.DataFrame(w.loc[regime.to_numpy()].to_numpy(), index=regime.index, columns=w.columns)
    return e.Book(panel[list(w.columns)], panel["rf"], positions)
