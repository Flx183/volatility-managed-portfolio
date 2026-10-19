"""
Phase 7, step 2: running a variant and measuring it
===================================================
    segments(book)                    the book cut into runs of consecutive holding periods
    prepare(book, forecaster)         the engine's stages 1-2 on each segment
    run(prepared, power, periods)     stages 3-4 on each segment. A segment, or its first periods,
                                      without a c_t holds cash and is not scored; each segment starts
                                      from cash, so its first trade pays for the whole entry
    downside(forecaster, freq)        the downside-volatility rule under any model, at monthly, weekly or
                                      daily origins
    month_of(res, periods)            the calendar month of each holding period (its closing trade)
    excess_months(res, periods)       monthly excess returns; a weekly run compounds its weeks
    metrics(res, periods)             Sharpe ratio, CER, Sortino, drawdown, turnover, break-even and
                                      the primary test, for the timed run and buy-and-hold
    summarize(m, r, break_even)       the same from monthly returns and holding periods already chosen
                                      (Phase 8 scores calendar months from its daily ledger)
    year_weighted(m, scale)           option (c) of Entry 10: each holding year's months scaled to an
                                      equal risk budget, both runs by the same number

Every statistic is computed on calendar-month excess returns, so monthly and weekly variants are
measured the same way (annualized with 12, Newey-West with 6 lags).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src import engine as e
from src.engine import evaluation as ev, execution
from src.engine.data import DAY, MONTH, WEEK
from src.frozen import freeze
from src.models import inputs, weekly
from src.walkforward import stats

GAMMA = 5                                 # risk aversion of every CER (Phase 6's)
MIN_PAIRS = freeze({MONTH: 120, WEEK: 520})   # c_t needs ten years of (signal, next-period return) pairs.
                                              # Daily books pass their own (Phase 8: robustness.checks.MIN_PAIRS)
DOWN_DAYS = 22                            # downside rule: the down-day share of the last 22 days' Σ r²


def segments(book: e.Book) -> list[e.Book]:
    """The book cut where its holding periods stop being consecutive (a year the screen left empty).
    A daily book counts trading days, so a weekend is no gap."""
    idx = book.positions.index
    step = book.calendar.get_indexer(idx) if book.freq == DAY else idx.asi8
    breaks = np.flatnonzero(np.diff(step) != 1) + 1
    return [e.Book(book.returns, book.rf, book.positions.iloc[s]) for s in np.split(np.arange(len(idx)), breaks)]


def prepare(book: e.Book, forecaster) -> list[tuple[e.Book, dict]]:
    return [(seg, e.prepare(seg, forecaster)) for seg in segments(book)]


def run(prepared: list[tuple[e.Book, dict]], power: float, periods: execution.Periods,
        max_size: float | None = e.ENGINE_RULES["max_size"], min_pairs: int | None = None,
        seed: int | None = None, c_window: int | None = None) -> pd.DataFrame:
    """The engine's output (decision, timed account, bench_ account) for every scored period, with
    `entry` marking each segment's first trade. seed shuffles each recipe's forecast (the placebo).
    c_window: c_t from the last c_window pairs instead of all of them (Phase 8's rolling windows)."""
    parts = []
    for seg, prep in prepared:
        n = min_pairs or MIN_PAIRS.get(seg.freq)
        if n is None:
            raise ValueError(f"Phase 7 fixed no c_t minimum for {seg.freq} books: pass min_pairs")
        try:
            dec = e.decide(seg, prep, power, max_size=max_size, min_months=n, seed=seed, c_window=c_window)
        except e.NoSize:                                     # no c_t anywhere in the segment: cash
            continue
        res = e.trade(seg, dec, periods)
        parts.append(res.assign(entry=res.index == res.index[0]))
    if not parts:
        raise e.NoSize("no holding period has a size")
    return pd.concat(parts)


def down_share(p: pd.Series, days: int = DOWN_DAYS) -> pd.Series:
    """Share of Σ r² over the last `days` trading days that comes from down days, at each day. If those
    days hold fewer than three down days, the window is extended back to 2 x `days`: Phase 2's rule for
    downside RV (Wang and Yan). Without it, a run of 22 up days gives a share of 0 and a forecast of 0."""
    p = p.dropna()
    r2, dn = p ** 2, (p < 0).astype(float)
    share = lambda n: r2.where(p < 0, 0.0).rolling(n).sum() / r2.rolling(n).sum()
    return share(days).where(dn.rolling(days).sum() >= 3, share(2 * days))


def daily_origins(p: pd.Series) -> pd.Series:
    """Position in p of every trading day, by day: a daily book's forecast origins."""
    p = p.dropna()
    return pd.Series(np.arange(len(p)), index=p.index.to_period(DAY))


ORIGINS = {MONTH: inputs.origins, WEEK: weekly.origins, DAY: daily_origins}


def downside(forecaster, freq: str = MONTH):
    """The downside-volatility rule under any model: σ̂⁻² = σ̂² x the down-day share of the last 22
    days' Σ r² (44 days if the 22 hold fewer than three down days), read at the forecast's own origins.
    Under M0 (monthly) this is the downside RV of the last 22 days."""
    if freq not in ORIGINS:
        raise ValueError(f"no forecast origins for frequency {freq!r}: monthly, weekly or daily")
    origins = ORIGINS[freq]

    def f(p: pd.Series) -> pd.Series:
        fc = forecaster(p)
        p = p.dropna()
        o = origins(p)
        share = pd.Series(down_share(p).to_numpy()[o.to_numpy()], index=o.index)
        if not fc.index.isin(share.index).all():
            raise ValueError(f"the forecast's periods are not the {freq} origins of its mix")
        return (fc * share.reindex(fc.index)).rename("forecast")
    f.__name__ = f"{getattr(forecaster, '__name__', 'forecast')}_down"
    return f


# ---------------------------------------------------------------- monthly view

def month_of(res: pd.DataFrame, periods: execution.Periods) -> pd.PeriodIndex:
    """The calendar month of each holding period's closing trade (a monthly period is its own month)."""
    if res.index.freqstr == MONTH:
        return res.index
    return pd.PeriodIndex(periods.closes.reindex(res.index).dt.to_period(MONTH))


def excess_months(res: pd.DataFrame, periods: execution.Periods, prefix: str = "", kappa: float | None = None) -> pd.Series:
    """Monthly excess returns: compounded net return minus compounded rf over the month's periods.
    kappa replaces the run's trading cost with κ per unit traded (for the break-even)."""
    net = res[f"{prefix}net"] if kappa is None else (1 - kappa * res[f"{prefix}turnover"]) * (1 + res[f"{prefix}pre_trade"]) - 1
    if res.index.freqstr == MONTH:
        return (net - res["rf"]).rename("excess")
    g = month_of(res, periods)
    grow = lambda x: np.expm1(np.log1p(x).groupby(g).sum())
    return (grow(net) - grow(res["rf"])).rename("excess")


def scored(res: pd.DataFrame, periods: execution.Periods, months: pd.PeriodIndex | None = None) -> pd.DataFrame:
    """Monthly excess returns of the timed run and buy-and-hold, and the mean size, on `months` (all
    scored months if None). A weekly run keeps only whole months: a month is scored or not."""
    m = pd.DataFrame({"excess": excess_months(res, periods), "bench_excess": excess_months(res, periods, "bench_"),
                      "size": res["size"].groupby(month_of(res, periods)).mean()})
    return m if months is None else m.loc[months]


def year_weighted(m: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Option (c) of PREREGISTRATION Entry 10: both runs' monthly excess returns multiplied by their holding
    year's scale (screened.year_scale), so every holding year carries about the same risk. The same number
    scales both runs in a year, so the test gains no look-ahead and timing no help. For a monthly book the
    holding year is the calendar year. Then ev.sharpe_gain_test(year_weighted(m, scale)) is the (c) test."""
    s = scale.reindex(m.index.year).to_numpy()
    if np.isnan(s).any():
        raise ValueError(f"no scale for the holding year of {m.index[np.isnan(s)][0]}")
    return m.assign(excess=m["excess"] * s, bench_excess=m["bench_excess"] * s)


def break_even(res: pd.DataFrame, periods: execution.Periods, months: pd.PeriodIndex, hi: float = 0.5) -> float:
    """One-way κ at which the timed Sharpe ratio falls to buy-and-hold's, on these months: 0 if timing
    does not beat buy-and-hold before trading costs, inf if the gain survives κ = hi."""
    def gap(k):
        x = excess_months(res, periods, kappa=k).loc[months]
        b = excess_months(res, periods, "bench_", kappa=k).loc[months]
        return ev.sharpe(x) - ev.sharpe(b)
    return solve_break_even(gap, hi)


def solve_break_even(gap, hi: float = 0.5) -> float:
    """The κ in [0, hi] where gap(κ), the Sharpe gain at κ, reaches 0: 0 if there is no gain at κ = 0,
    inf if it survives κ = hi."""
    if gap(0.0) <= 0:
        return 0.0
    if gap(hi) > 0:
        return np.inf
    return brentq(gap, 0.0, hi, xtol=1e-10)


def metrics(res: pd.DataFrame, periods: execution.Periods, months: pd.PeriodIndex | None = None,
            gamma: float = GAMMA) -> dict:
    """Everything the results table reports for one run, on calendar months (all its scored months
    if `months` is None). Turnover leaves out each segment's entry trade."""
    m = scored(res, periods, months)
    r = res[np.asarray(month_of(res, periods).isin(m.index))]
    return summarize(m, r, break_even(res, periods, m.index), gamma)


def summarize(m: pd.DataFrame, r: pd.DataFrame, break_even: float, gamma: float = GAMMA) -> dict:
    """metrics() from its parts: m, the monthly excess returns (excess, bench_excess, size) on the scored
    months; r, the holding periods those months hold (size, capped, turnover, bench_turnover, entry); and
    the break-even κ on those months."""
    test = ev.sharpe_gain_test(m)
    x, b = m["excess"], m["bench_excess"]
    years = len(m) / 12
    rebal = r.loc[~r["entry"]]
    out = {"months": len(m), "first": str(m.index[0]), "last": str(m.index[-1]),
           "sharpe": ev.sharpe(x), "bh_sharpe": ev.sharpe(b),
           **{k: test[k] for k in ("gain", "se", "z", "p", "lower", "upper", "mde", "reading")},
           "cer": ev.cer(x, gamma), "bh_cer": ev.cer(b, gamma), "cer_gain": ev.cer(x, gamma) - ev.cer(b, gamma),
           "sortino": stats.sortino(x), "bh_sortino": stats.sortino(b),
           "max_drawdown": ev.max_drawdown(x), "bh_max_drawdown": ev.max_drawdown(b),
           "volatility": x.std() * np.sqrt(12), "bh_volatility": b.std() * np.sqrt(12),
           "mean": x.mean() * 12, "bh_mean": b.mean() * 12,
           "mean_size": r["size"].mean(), "share_capped": r["capped"].mean(),
           "turnover_per_rebalance": rebal["turnover"].mean(), "turnover_a_year": rebal["turnover"].sum() / years,
           "bh_turnover_a_year": rebal["bench_turnover"].sum() / years,
           "break_even": break_even}
    span = ev.spanning(x, b)
    out.update({"alpha": span["alpha"], "t_alpha": span["t_alpha"]})
    return out
