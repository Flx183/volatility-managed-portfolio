"""
Track A, sizing: the variants' sizes
====================================
Each variant starts from the frozen engine's decision for a holding month (e.decide: the forecast made at the
close before it, the signal, c_t and the capped size) and sets the size it holds:

    A0   the frozen size, c_t x forecast⁻¹ capped at 1.5: unchanged
    A1   σ̂_y √(22/252) / √forecast, capped at 1.5. σ̂_y is the book's buy-and-hold volatility (a year) over the
         10-year window set at the last year-end, read over the forecast's 22 days. So the target moves once a
         year with the book's own history, and the mean size sits near 1 whatever c_t has learned

    sigma_screened(panel, rec, mix)       σ̂_y of a screened book: Entry 10's, over its recipe's window
    sigma_trailing(book)                  σ̂_y of a book with fixed positions (SPY alone): the last 10 years up to
                                          each December year-end
    resize(dec, variant, sigma)           the decision table with the variant's size
    run(prepared, periods, variant, ...)  Phase 7's segment runner (runs.run) with the variant's size

A1 holds in the months A0 holds (those with a c_t), so both are scored on the same months. Every size is known at
the close before its month: σ̂_y at the year-end, the forecast and c_t at the month-end. (Entry 20 dropped A2, the
fixed blend: it lives on as Track B's B1.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import engine as e
from src.engine import execution
from src.engine.data import MONTH
from src.overlay import TRACK_A
from src.screen import year_ends
from src.walkforward import runs, screened

VARIANTS = tuple(TRACK_A["variants"])


def sigma_screened(panel: pd.DataFrame, rec: pd.DataFrame, mix: str) -> pd.Series:
    """σ̂_y by holding year for a screened book: 1 / screened.year_scale, the volatility option (c) scales by."""
    return (1 / screened.year_scale(panel, rec, mix)).rename("sigma")


def sigma_trailing(book: e.Book, years: int = TRACK_A["window_years"]) -> pd.Series:
    """σ̂_y by holding year for a book whose positions never change: √252 x the daily standard deviation of the
    book at size 1 over the `years` years up to each December year-end (all of its history if shorter)."""
    pos = book.positions
    if (pos.nunique() > 1).any():
        raise ValueError("sigma_trailing needs fixed positions; a screened book uses sigma_screened")
    w = pos.iloc[0]
    p = book.returns[w.index].dropna() @ w
    ends = year_ends(p.index)
    out = {}
    for d in ends[ends.month == 12]:
        win = p.loc[(p.index > d - pd.DateOffset(months=12 * years)) & (p.index <= d)]
        out[d.year + 1] = win.std() * np.sqrt(252)
    return pd.Series(out, name="sigma").rename_axis("year")


def resize(dec: pd.DataFrame, variant: str, sigma: pd.Series | None = None, max_size: float | None = TRACK_A["cap"],
           horizon: int = TRACK_A["horizon"]) -> pd.DataFrame:
    if variant == "A0":
        return dec
    if variant != "A1":
        raise ValueError(f"unknown variant {variant!r}: one of {VARIANTS}")
    if dec.index.freqstr != MONTH or sigma is None:
        raise ValueError("A1 sizes monthly holding periods and needs σ̂_y by holding year")
    s = sigma.reindex(dec.index.year).to_numpy()
    if np.isnan(s).any():
        raise ValueError(f"no σ̂_y for the holding year of {dec.index[np.isnan(s)][0]}")
    if (dec["forecast"] <= 0).any():
        raise ValueError("variance forecasts must be positive")
    c = pd.Series(s * np.sqrt(horizon / 252), index=dec.index)
    signal = dec["forecast"] ** -0.5
    raw = c * signal
    cap = np.inf if max_size is None else max_size
    return dec.assign(signal=signal, c=c, size=raw.clip(lower=0, upper=cap), capped=raw > cap)


def run(prepared: list[tuple[e.Book, dict]], periods: execution.Periods, variant: str, sigma: pd.Series | None = None,
        power: float = 1.0, max_size: float | None = TRACK_A["cap"], min_pairs: int | None = None) -> pd.DataFrame:
    """runs.run with the variant's size: A0's decision for every period (the frozen rule, its c_t and cap), then
    resize, then the trade. A0 reproduces runs.run to the last digit."""
    parts = []
    for seg, prep in prepared:
        n = min_pairs or runs.MIN_PAIRS.get(seg.freq)
        if n is None:
            raise ValueError(f"Phase 7 fixed no c_t minimum for {seg.freq} books: pass min_pairs")
        try:
            dec = e.decide(seg, prep, power, max_size=max_size, min_months=n)
        except e.NoSize:
            continue
        res = e.trade(seg, resize(dec, variant, sigma, max_size), periods)
        parts.append(res.assign(entry=res.index == res.index[0]))
    if not parts:
        raise e.NoSize("no holding period has a size")
    return pd.concat(parts)
