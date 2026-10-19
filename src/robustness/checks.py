"""
Phase 8, the checks
===================
    Spec, BASE                                one specification: the base, or the base with one thing changed
    variation(base, **change), spec_of(change)   a one-thing change, from keywords or from one of CHECKS'
    resolve(spec)                             everything a run needs from its spec: forecaster, power, cap,
                                              c_t minimum and window. run and time_each read only this
    recipes(panel, family, survivors, drop)   the recipes of a book family, with one asset dropped or none:
                                              "screened" (the primary's real-time screen, Phase 7) or
                                              "fixed" (Phase 4's paper recipe on market, momentum, BAB)
    book(panel, rec, freq)                    the inverse-volatility book of those recipes, held monthly,
                                              weekly or daily
    variations(assets)                        every one-at-a-time change, check by check
    run(panel, rec, spec)                     one spec through the engine
    asset_sizes(book, ...)                    each asset timed alone: its size and whether its cap binds
    time_each(book, periods, ...)             each asset timed alone, then combined at the mix's weights
    scored(book, res, periods, months)        calendar-month excess returns, from the daily ledger
    whole_months(book, res)                   the calendar months a run holds from first day to last
    row(book, res, periods, months)           the metrics a robustness row reports, on those months
    effects(base, var, base_all)              the change in gain, split: the spec effect on the same months
                                              (with its paired bootstrap interval) and the sample effect
    verdict(gains, readings)                  "holds", "sign holds" or "flips", against the base
    placebo_shares(draws, real, months)       the placebo rule read on a set of months
    risk_shares(panel, rec, method)           each asset's share of buy-and-hold's variance

The base is the frozen specification (Entry 6): inverse-volatility mix, M3 GJR-GARCH, inverse-variance
timing, monthly, cap 1.5, real-time c_t from 120 pairs, no costs. A change touches one of these and
nothing else (PREREGISTRATION Entry 8); Spec enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import engine as e
from src import models as md
from src import portfolio as pf
from src import walkforward as wf
from src.config import FACTORS
from src.engine import evaluation as ev, execution
from src.engine.data import DAY, MONTH
from src.frozen import freeze
from src.robustness import CHECKS, ROBUST_RULES, ledger
from src.walkforward import runs, screened, stats

FIXED = pf.TRACKS["paper"]["assets"]          # market, momentum, BAB
METHOD = pf.PRIMARY[1]                        # the inverse-volatility mix
MAX_SIZE = e.ENGINE_RULES["max_size"]
# c_t minimums: Phase 7's monthly and weekly (Entry 6), and Phase 8's daily (Entry 8). A new table: Phase 7's is frozen.
MIN_PAIRS = freeze({**wf.WF_RULES["min_pairs"], DAY: ROBUST_RULES["daily_min_pairs"]})


# ---------------------------------------------------------------- the specification

@dataclass(frozen=True)
class Spec:
    order: str = "combined"                   # "combined" (combine, then time) or "each" (time each, then combine)
    rule: str = "inv_var"                     # a key of walkforward.RULES
    window: str = "expanding"                 # or a key of ROBUST_RULES["windows"]: the GARCH days and c_t pairs
    freq: str = MONTH                         # MONTH, WEEK or DAY
    cap: float | None = MAX_SIZE              # the leverage cap; None: no cap
    costs: tuple[float, float] = (0.0, 0.0)   # κ one-way on every instrument (both runs), and the borrow spread
    drop: str | None = None                   # an asset left out of every recipe (the caller passes those recipes)
    era: str | None = None                    # one era's months (the caller scores those months)


BASE = Spec()
KEYS = {"freq": "freq", "window": "window", "max_size": "cap", "rule": "rule", "order": "order", "drop": "drop",
        "era": "era"}                                                    # CHECKS' names -> Spec's


def changes(spec: Spec) -> dict:
    """The fields where spec differs from the base."""
    return {f.name: getattr(spec, f.name) for f in fields(Spec) if getattr(spec, f.name) != getattr(BASE, f.name)}


def variation(base: Spec = BASE, **change) -> Spec:
    if len(change) != 1:
        raise ValueError(f"Phase 8 changes one thing at a time, not {len(change)}: {sorted(change)}")
    return replace(base, **change)


def spec_of(change: dict) -> Spec:
    """The Spec of one of CHECKS' changes ({} is the base). κ and its borrowing are one change: the costs."""
    if not change:
        return BASE
    out = {KEYS[k]: v for k, v in change.items() if k not in ("kappa", "borrow")}
    if "kappa" in change or "borrow" in change:
        out["costs"] = (change.get("kappa", 0.0), change.get("borrow", 0.0))
    return variation(BASE, **out)


def resolve(spec: Spec, frozen=None) -> dict:
    """What a run needs from its spec, and nothing from module defaults: the forecaster (M3 under the spec's
    frequency and estimation window; the downside rule wraps it), the power of the signal, the cap, the c_t
    minimum and the c_t window. frozen: the forecaster factory, md.frozen unless a test passes its own."""
    if len(changes(spec)) > 1:
        raise ValueError(f"Phase 8 changes one thing at a time, not {sorted(changes(spec))}")
    if spec.order not in ("combined", "each"):
        raise ValueError(f"order is 'combined' or 'each', not {spec.order!r}")
    days, pairs = (None, None) if spec.window == "expanding" else ROBUST_RULES["windows"][spec.window]
    power, down = wf.RULES[spec.rule]
    f = (frozen or md.frozen)(True, spec.freq, days)
    if down:
        f = runs.downside(f, spec.freq)
    return {"forecaster": f, "power": power, "max_size": spec.cap, "min_pairs": MIN_PAIRS[spec.freq], "c_window": pairs}


def costs_for(book: e.Book, spec: Spec) -> e.Costs:
    """κ on every instrument the book holds (both runs pay it), and borrowed cash at rf + spread."""
    kappa, borrow = spec.costs
    if not kappa and not borrow:
        return e.NO_COSTS
    return e.Costs(kappa={c: kappa for c in book.positions.columns}, borrow_spread=borrow)


# ---------------------------------------------------------------- books

def recipes(panel: pd.DataFrame, family: str, survivors: pd.Series | None = None, drop: str | None = None) -> pd.DataFrame:
    """The recipes at every year-end, without `drop`. "screened": the real-time screen's survivors at
    each year-end (a year left empty is cash). "fixed": Phase 4's paper recipe on the fixed assets
    (dropping BAB gives Phase 4's paper market + momentum track, the ETF track's paper twin)."""
    if family == "screened":
        if survivors is None:
            raise ValueError("the screened family needs the screen's survivors: screened.survivors(screened.screen_all(...))")
        return screened.recipes(panel, survivors.map(lambda names: [a for a in names if a != drop]))
    if family != "fixed":
        raise ValueError(f"the families are 'screened' and 'fixed', not {family!r}")
    if drop is None:
        return pf.recipes(panel, "paper")
    spec = {**pf.TRACKS["paper"], "assets": [a for a in FIXED if a != drop]}
    return pf.recipes(panel, spec)


def book(panel: pd.DataFrame, rec: pd.DataFrame, freq: str = MONTH) -> e.Book:
    """The inverse-volatility book: each recipe from the first rebalance after its year-end (Phase 7).
    Monthly, this is e.data.from_phase4's book for Phase 4's recipes."""
    return screened.book(panel, rec, METHOD, freq)


def held_assets(rec: pd.DataFrame) -> list[str]:
    """The factors any recipe holds, in FACTORS order."""
    w = rec.filter(regex="^w_")
    return [f for f in FACTORS if f"w_{f}" in w and (w[f"w_{f}"] != 0).any()]


# ---------------------------------------------------------------- the variations

NAMES = {"mkt": "market", "mom": "momentum", "rmw": "profitability", "bab": "BAB", "smb": "size", "hml": "value",
         "cma": "investment"}


def variations(assets: list[str]) -> list[tuple[str, str, dict]]:
    """(check, label, change) for every check, the base's own setting first with an empty change.
    Leave-one-out drops each of `assets` in turn. spec_of(change) is its Spec."""
    out = []
    for check, (base, changes_) in CHECKS.items():
        if check == "leave_one_out":
            changes_ = [(f"drop {NAMES[a]}", {"drop": a}) for a in assets]
        out.append((check, base, {}))
        out += [(check, label, change) for label, change in changes_]
    return out


def run(panel: pd.DataFrame, rec: pd.DataFrame, spec: Spec | dict = BASE, frozen=None) -> tuple[e.Book, pd.DataFrame, execution.Periods]:
    """One spec (or one of CHECKS' changes) through the engine: the book, the run (one row per scored
    holding period, as runs.run) and its periods. rec are the recipes of spec.drop; spec.era is the caller's
    too (other months)."""
    spec = spec if isinstance(spec, Spec) else spec_of(spec)
    cfg = resolve(spec, frozen)
    b = book(panel, rec, spec.freq)
    periods = execution.holding_periods(b, costs_for(b, spec))
    if spec.order == "each":
        return b, time_each(b, periods, **cfg), periods
    return b, runs.run(runs.prepare(b, cfg["forecaster"]), cfg["power"], periods, max_size=cfg["max_size"],
                       min_pairs=cfg["min_pairs"], c_window=cfg["c_window"]), periods


# ---------------------------------------------------------------- time each, then combine

def asset_sizes(book: e.Book, forecaster, power: float, max_size: float | None, min_pairs: int,
                c_window: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each asset the book holds, timed alone on its own daily returns: its size, min(c_i,t x σ̂_i,t^(-2 power),
    cap), and whether the cap binds, in every holding period of the book. NaN and False where it has no
    size; an asset with too little history for its c_t has none anywhere."""
    idx = book.positions.index
    S, C = {}, {}
    for a in [a for a in book.positions.columns if (book.positions[a] != 0).any()]:
        single = e.Book.single(book.returns[a], book.rf, a, book.freq)
        try:
            dec = e.decide(single, e.prepare(single, forecaster), power, max_size=max_size, min_months=min_pairs,
                           c_window=c_window)
            S[a], C[a] = dec["size"].reindex(idx), dec["capped"].reindex(idx, fill_value=False)
        except e.NoSize:
            S[a], C[a] = pd.Series(np.nan, idx), pd.Series(False, idx)
    return pd.DataFrame(S, index=idx), pd.DataFrame(C, index=idx)


def time_each(book: e.Book, periods: execution.Periods, *, forecaster, power: float, max_size: float | None,
              min_pairs: int, c_window: int | None) -> pd.DataFrame:
    """Time each asset alone, then combine: asset i's position is w_i x size_i (asset_sizes). The benchmark
    is the book at size 1, as always. As in runs.run, a segment's periods before every asset it holds has a
    size are cash and not scored, and so is a segment where that never happens (an asset too short for its
    c_t). Columns as runs.run; `size` is Σ w_i size_i (the weights sum to 1), `capped` whether any asset is.
    Every argument comes from resolve(spec)."""
    S, C = asset_sizes(book, forecaster, power, max_size, min_pairs, c_window)
    parts = []
    for seg in runs.segments(book):
        pos = seg.positions[list(S.columns)]
        need = pos != 0
        ok = (S.loc[pos.index].notna() | ~need).all(axis=1)
        if not ok.any():
            continue
        start = ok.idxmax()
        if not ok.loc[start:].all():
            raise ValueError(f"an asset loses its size inside a segment, at {ok.loc[start:].idxmin()}")
        pos, need = pos.loc[start:], need.loc[start:]
        s = S.loc[pos.index].where(need, 0.0)
        timed = execution.simulate(periods, pos * s)
        bench = execution.simulate(periods, pos).add_prefix("bench_").drop(columns="bench_rf")
        dec = pd.DataFrame({"size": (pos * s).sum(axis=1), "capped": (C.loc[pos.index] & need).any(axis=1)})
        res = pd.concat([dec, timed, bench], axis=1)
        parts.append(res.assign(entry=res.index == res.index[0]))
    if not parts:
        raise e.NoSize("no holding period has a size for every asset")
    return pd.concat(parts)


# ---------------------------------------------------------------- scoring

def whole_months(book: e.Book, res: pd.DataFrame) -> pd.PeriodIndex:
    """The calendar months whose every trading day the run holds a position (ledger.whole). Only these are
    scored: a weekly or daily run can start or stop inside a month."""
    return ledger.whole(book, res)


def scored(book: e.Book, res: pd.DataFrame, periods: execution.Periods, months: pd.PeriodIndex | None = None) -> pd.DataFrame:
    """Calendar-month excess returns of the timed run and buy-and-hold, and the mean size held, from the
    daily ledger: on `months`, each of which the run must hold on every trading day, or on all such months."""
    m = ledger.months(book, res, periods)
    m = m[m["whole"]]
    if months is not None:
        missing = months.difference(m.index)
        if len(missing):
            raise ValueError(f"{missing[0]} is not held on every trading day: it cannot be scored")
        m = m.loc[months]
    return m[["excess", "bench_excess", "size"]]


def row(book: e.Book, res: pd.DataFrame, periods: execution.Periods, months: pd.PeriodIndex) -> dict:
    """runs.summarize on these months, from the daily ledger: every statistic on calendar-month excess
    returns, the break-even on the same months; size, capping and turnover over the periods whose opening
    trade the months book (a period's first day). An era with fewer than min_era_months is reported
    without a reading."""
    if len(months) < 3:
        return {"months": len(months)}
    P = ledger.pieces(book, res, periods)
    m = ledger.monthly(P, res).loc[months]
    if not months.isin(ledger.whole(book, res)).all():
        raise ValueError("a scored month must be held on every trading day")
    opens = P.loc[P["opens"]].set_index("period")["month"]
    r = res[opens.reindex(res.index).isin(months).to_numpy()]
    gap = lambda k: (lambda x: ev.sharpe(x["excess"]) - ev.sharpe(x["bench_excess"]))(ledger.monthly(P, res, k).loc[months])
    out = runs.summarize(m, r, runs.solve_break_even(gap))
    if len(months) < ROBUST_RULES["min_era_months"]:
        out["reading"] = "too short to read"
    return out


@lru_cache(maxsize=2)
def _counts(n: int, block: float) -> np.ndarray:
    return stats.counts(n, block)


def effects(base: pd.DataFrame, var: pd.DataFrame, base_all: pd.DataFrame, level: float = 0.95) -> dict:
    """A variation's change in Sharpe gain, split in two. base and var: the monthly excess returns (excess,
    bench_excess) of the base run and the variation on the same months, the months both score; base_all:
    the base on all its months.

        spec effect    gain(variation) - gain(base), on the same months: what the change does
        sample effect  gain(base on those months) - gain(base on all its months): what losing months does

    The two sum to the variation's gain minus the base's. The spec effect's standard error is the stationary
    bootstrap's with both runs resampled on the same months (paired: they move together), the larger of
    Phase 7's mean blocks of 6 and 12, and its interval spec ± z se, two-sided at `level`."""
    if not base.index.equals(var.index):
        raise ValueError("the base and the variation must be scored on the same months")
    gain = lambda m: ev.sharpe(m["excess"]) - ev.sharpe(m["bench_excess"])
    spec = gain(var) - gain(base)
    cols = ["excess", "bench_excess"]
    se = 0.0
    for blk in stats.BOOT["blocks"]:
        C = _counts(len(base), blk)
        v, b = stats.sharpe_draws(C, var[cols].to_numpy()), stats.sharpe_draws(C, base[cols].to_numpy())
        se = max(se, float(np.std((v[:, 0] - v[:, 1]) - (b[:, 0] - b[:, 1]), ddof=1)))
    z = norm.ppf(0.5 + level / 2)
    return {"spec_effect": spec, "spec_se": se, "spec_lower": spec - z * se, "spec_upper": spec + z * se,
            "sample_effect": gain(base) - gain(base_all)}


def verdict(base_gain: float, base_reading: str, gains: list[float], readings: list[str]) -> str:
    """Against the base: "flips" if some gain has the strictly opposite sign; else "sign holds" if some
    reading differs; else "holds". A gain of exactly 0, or a base gain of 0, has no sign: a tie, not a flip."""
    if any(g * base_gain < 0 for g in gains):
        return "flips"
    if any(r != base_reading for r in readings):
        return "sign holds"
    return "holds"


# ---------------------------------------------------------------- the placebo

def placebo_shares(draws: pd.DataFrame, real: pd.DataFrame, months: pd.PeriodIndex) -> dict:
    """On these months: the share of shuffled spanning α's the real α beats (the placebo rule, pass at
    95%), and the share of shuffled Sharpe gains the real gain beats (information). draws: walkforward.build
    .placebo's, each draw's timed monthly excess."""
    x, b = real.loc[months, "excess"], real.loc[months, "bench_excess"]
    D = draws.loc[:, months]
    alphas = np.array([ev.alpha(D.loc[i], b) for i in D.index])
    gains = np.array([ev.sharpe(D.loc[i]) for i in D.index]) - ev.sharpe(b)
    a = ev.alpha(x, b)
    return {"months": len(months), "alpha": a, "alpha_share": ev.placebo_share(a, alphas),
            "pass": bool(ev.placebo_share(a, alphas) >= e.ENGINE_RULES["placebo_share"]),
            "gain": ev.sharpe(x) - ev.sharpe(b), "gain_share": ev.placebo_share(ev.sharpe(x) - ev.sharpe(b), gains)}


# ---------------------------------------------------------------- information beside leave-one-out

def risk_shares(panel: pd.DataFrame, rec: pd.DataFrame, method: str) -> pd.DataFrame:
    """Each asset's share of the mix's variance, w_i (Σw)_i / w'Σw, at every year-end, with Σ the
    covariance of daily returns over the recipe's own 10-year window (the one its weights came from).
    The shares sum to 1; a negative share is a hedge."""
    w = rec.xs(method, level="method").filter(regex="^w_")
    w.columns = w.columns.str.removeprefix("w_")
    starts = rec.xs(method, level="method")["window_start"]
    rows = {}
    for asof, wt in w.iterrows():
        names = wt.index[wt != 0]
        cov = panel.loc[pd.Timestamp(starts.loc[asof]):asof, names].cov().to_numpy()
        x = wt[names].to_numpy()
        rows[asof] = pd.Series(x * (cov @ x) / (x @ cov @ x), names)
    return pd.DataFrame(rows).T.reindex(columns=[f for f in FACTORS if f in w.columns]).fillna(0.0)
