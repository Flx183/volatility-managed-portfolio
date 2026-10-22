"""
Track B, rules: the variants' sizes
===================================
Every variant starts from the engine's decision for a holding month (e.decide: the forecast made at the close
before it, c_t, and the size c_t x forecast^-power, capped at 1.5) and sets the size it holds:

    frozen   the frozen size, c_t / forecast variance (power 1), unchanged (the reference)
    B1, B2   0.5 + 0.5 x the inverse-volatility size (power 0.5, its own c_t, capped at 1.5). B2 runs on the downside
             forecast (runs.downside): Phase 7's downside-volatility rule, blended
    B3       1, unless the forecast is extreme (signals: rank); then the frozen size (power 1)
    B4       the larger of 1 and the frozen size if the book is above its trend (signals: above); else the frozen size

    signals(tab, min_months)             at each month-end of a recipe's table: the forecast's rank among its last
                                         120 values, whether it is extreme, and whether the book is above its trend
    resize(dec, variant, sig)            the decision table with the variant's size; sig is signals() at each
                                         holding month's decision close
    run(prepared, periods, variant, ...)  Phase 7's segment runner (runs.run) with the variant's size

Both signals read a recipe's own table (prepare's: the mix held next month, back-cast over its whole history, as
the forecast and c_t are), and only its rows up to the decision close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import engine as e
from src.engine import execution
from src.engine.data import MONTH
from src.fixes import TRACK_B
from src.walkforward import runs

VARIANTS = ("frozen", *TRACK_B["family"])


def signals(tab: pd.DataFrame, min_months: int, window: int = TRACK_B["window"], q: float = TRACK_B["extreme"],
            trend: int = TRACK_B["trend_months"]) -> pd.DataFrame:
    """tab: a recipe's monthly table (forecast, ret, rf). rank: the share of the last `window` forecasts (this one
    included) at or below this one, once min_months exist; extreme: rank in the top or bottom q. above: the book's
    value at size 1, Π(1 + ret + rf), above the mean of its last `trend` month-end values."""
    f = tab["forecast"].dropna()
    rank = f.rolling(window, min_periods=min_months).apply(lambda w: (w <= w[-1]).mean(), raw=True)
    level = (1 + tab["ret"] + tab["rf"]).cumprod()
    mean = level.rolling(trend).mean()
    return pd.DataFrame({"rank": rank, "extreme": ((rank > 1 - q) | (rank <= q)).where(rank.notna()),
                         "above": (level > mean).where(mean.notna())})


def decision_signals(dec: pd.DataFrame, prep: dict[int, pd.DataFrame], min_months: int) -> pd.DataFrame:
    """signals() of each holding month's recipe, at its decision close (the month before)."""
    if dec.index.freqstr != MONTH:
        raise ValueError("Track B sizes monthly holding periods")
    out = pd.DataFrame(index=dec.index, columns=["rank", "extreme", "above"], dtype=object)
    for k, tab in prep.items():
        held = dec.index[dec["recipe"] == k]
        if len(held):
            out.loc[held] = signals(tab, min_months).reindex(held - 1).to_numpy()
    return out


def resize(dec: pd.DataFrame, variant: str, sig: pd.DataFrame | None = None) -> pd.DataFrame:
    s = dec["size"]
    if variant == "frozen":
        return dec
    if variant in ("B1", "B2"):
        w = TRACK_B["blend"]
        return dec.assign(size=w + (1 - w) * s)
    if variant not in ("B3", "B4"):
        raise ValueError(f"unknown variant {variant!r}: one of {VARIANTS}")
    col = "extreme" if variant == "B3" else "above"
    flag = sig[col].reindex(dec.index)
    if flag.isna().any():
        raise ValueError(f"{variant}: no '{col}' signal for {flag.index[flag.isna()][0]}")
    flag = flag.astype(bool)
    if variant == "B3":
        return dec.assign(size=s.where(flag, 1.0), capped=dec["capped"] & flag)
    return dec.assign(size=np.maximum(s, 1.0).where(flag, s))


def run(prepared: list[tuple[e.Book, dict]], periods: execution.Periods, variant: str,
        max_size: float | None = TRACK_B["cap"], min_pairs: int | None = None) -> pd.DataFrame:
    """runs.run with the variant's size: the decision at the variant's power (TRACK_B["power"]) for every period,
    then resize, then the trade. The signals' rank needs as many forecasts as c_t needs pairs, so every variant
    holds in the frozen run's months. "frozen" reproduces runs.run at power 1 to the last digit."""
    power = TRACK_B["power"][variant]
    parts = []
    for seg, prep in prepared:
        n = min_pairs or runs.MIN_PAIRS.get(seg.freq)
        if n is None:
            raise ValueError(f"Phase 7 fixed no c_t minimum for {seg.freq} books: pass min_pairs")
        try:
            dec = e.decide(seg, prep, power, max_size=max_size, min_months=n)
        except e.NoSize:
            continue
        sig = decision_signals(dec, prep, n) if variant in ("B3", "B4") else None
        res = e.trade(seg, resize(dec, variant, sig), periods)
        parts.append(res.assign(entry=res.index == res.index[0]))
    if not parts:
        raise e.NoSize("no holding period has a size")
    return pd.concat(parts)
