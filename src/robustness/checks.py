"""
Phase 8, the checks
===================
    recipes(panel, family, survivors, drop)   the recipes of a book family, with one asset dropped or none:
                                              "screened" (the primary's real-time screen, Phase 7) or
                                              "fixed" (Phase 4's paper recipe on market, momentum, BAB)
    book(panel, rec, freq)                    the inverse-volatility book of those recipes, held monthly,
                                              weekly or daily
    variations(assets)                        every one-at-a-time change, check by check
    run(panel, rec, change)                   one change through the engine
    time_each(book, forecaster, periods)      each asset timed alone, then combined at the mix's weights
    whole_months(book, res)                   the calendar months a run holds from first day to last
    row(res, periods, months)                 the metrics a robustness row reports, on those months
    verdict(gains, readings)                  "holds", "sign holds" or "flips", against the base
    placebo(book, forecaster, periods)        the timed monthly excess of 1,000 shuffled-forecast runs
    risk_shares(panel, rec, method)           each asset's share of buy-and-hold's variance

The base is the frozen specification (Entry 6): inverse-volatility mix, M3 GJR-GARCH, inverse-variance
timing, monthly, cap 1.5, real-time c_t from 120 pairs, no costs. A change touches one of these and
nothing else (PREREGISTRATION Entry 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import engine as e
from src import models as md
from src import portfolio as pf
from src import walkforward as wf
from src.config import FACTORS
from src.engine import evaluation as ev, execution
from src.engine.data import MONTH
from src.robustness import CHECKS, ROBUST_RULES
from src.walkforward import runs, screened

FIXED = pf.TRACKS["paper"]["assets"]          # market, momentum, BAB
METHOD = pf.PRIMARY[1]                        # the inverse-volatility mix
MAX_SIZE = e.ENGINE_RULES["max_size"]


# ---------------------------------------------------------------- books

def recipes(panel: pd.DataFrame, family: str, survivors: pd.Series | None = None, drop: str | None = None) -> pd.DataFrame:
    """The recipes at every year-end, without `drop`. "screened": the real-time screen's survivors at
    each year-end (a year left empty is cash). "fixed": Phase 4's paper recipe on the fixed assets
    (dropping BAB gives Phase 4's paper market + momentum track, the ETF track's paper twin)."""
    if family == "screened":
        return screened.recipes(panel, survivors.map(lambda names: [a for a in names if a != drop]))
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
    Leave-one-out drops each of `assets` in turn."""
    out = []
    for check, (base, changes) in CHECKS.items():
        if check == "leave_one_out":
            changes = [(f"drop {NAMES[a]}", {"drop": a}) for a in assets]
        out.append((check, base, {}))
        out += [(check, label, change) for label, change in changes]
    return out


def costs_for(book: e.Book, change: dict) -> e.Costs:
    """κ on every instrument the book holds (both runs pay it), and borrowed cash at rf + spread."""
    if "kappa" not in change:
        return e.NO_COSTS
    return e.Costs(kappa={c: change["kappa"] for c in book.positions.columns}, borrow_spread=change.get("borrow", 0.0))


def run(panel: pd.DataFrame, rec: pd.DataFrame, change: dict) -> tuple[e.Book, pd.DataFrame, execution.Periods]:
    """One change from the base through the engine: the book, the run (one row per holding period, as
    runs.run) and its periods. A change names at most one of: freq, window, max_size, rule, order,
    kappa (with borrow). drop and era are handled by the caller (other recipes; other months)."""
    freq = change.get("freq", MONTH)
    b = book(panel, rec, freq)
    periods = execution.holding_periods(b, costs_for(b, change))
    days, pairs = ROBUST_RULES["windows"][change["window"]] if "window" in change else (None, None)
    rule = change.get("rule", "inv_var")
    power, down = wf.RULES[rule]
    f = md.frozen(True, freq, days)
    if down:
        f = runs.downside(f, freq)
    max_size = change["max_size"] if "max_size" in change else MAX_SIZE
    if change.get("order") == "each":
        return b, time_each(b, f, periods, max_size), periods
    return b, runs.run(runs.prepare(b, f), power, periods, max_size=max_size, c_window=pairs), periods


def time_each(book: e.Book, forecaster, periods: execution.Periods, max_size: float | None = MAX_SIZE,
              min_pairs: int | None = None) -> pd.DataFrame:
    """Time each asset alone, then combine. Asset i gets its own size, min(c_i,t / σ̂²_i,t, cap), from the
    forecast of its own daily returns and its own real-time c_i,t; its position is w_i x size_i. The
    benchmark is the book at size 1, as always. A segment starts once every asset it holds has a size.
    Columns as runs.run; `size` is Σ w_i size_i (the weights sum to 1), `capped` whether any asset is."""
    held = [a for a in book.positions.columns if (book.positions[a] != 0).any()]
    n = min_pairs or runs.MIN_PAIRS[book.freq]
    sizes, capped = {}, {}
    for a in held:
        single = e.Book.single(book.returns[a], book.rf, a, book.freq)
        dec = e.decide(single, e.prepare(single, forecaster), 1.0, max_size=max_size, min_months=n)
        sizes[a], capped[a] = dec["size"], dec["capped"]
    S = pd.DataFrame(sizes).reindex(book.positions.index)
    C = pd.DataFrame(capped).astype(bool).reindex(book.positions.index, fill_value=False)
    parts = []
    for seg in runs.segments(book):
        pos = seg.positions[held]
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
    """The calendar months whose every trading day the run holds a position. A monthly run holds whole
    months; a weekly or daily run can start or stop inside one (its week or day straddles it), and such a
    month is not scored."""
    labels = pd.Series(execution.period_labels(book.returns.index, 0, book.freq), index=book.returns.index)
    held = labels.isin(res.index)
    whole = held.groupby(book.returns.index.to_period(MONTH)).all()
    return pd.PeriodIndex(whole.index[whole.to_numpy()], freq=MONTH)


def row(res: pd.DataFrame, periods: execution.Periods, months: pd.PeriodIndex) -> dict:
    """runs.metrics on these months (every statistic on calendar-month excess returns). An era with
    fewer than min_era_months is reported without a reading."""
    if len(months) < 3:
        return {"months": len(months)}
    out = runs.metrics(res, periods, months)
    if len(months) < ROBUST_RULES["min_era_months"]:
        out["reading"] = "too short to read"
    return out


def verdict(base_gain: float, base_reading: str, gains: list[float], readings: list[str]) -> str:
    """Against the base: "holds" if every variation has the base's reading; "sign holds" if some reading
    differs but every gain has the base's sign; "flips" if some gain has the other sign."""
    if any(np.sign(g) != np.sign(base_gain) for g in gains):
        return "flips"
    if any(r != base_reading for r in readings):
        return "sign holds"
    return "holds"


# ---------------------------------------------------------------- the placebo

def placebo(book: e.Book, forecaster, periods: execution.Periods, draws: int = ROBUST_RULES["placebo_draws"],
            jobs: int = -2) -> pd.DataFrame:
    """The engine's placebo on the walk-forward (Phase 7's): each draw shuffles every recipe's forecast in
    time and reruns sizes (c_t included) and trading. Returns the timed monthly excess return of each
    draw (rows: draws 0..draws-1, the seeds; columns: months). Buy-and-hold does not depend on the
    forecast, so the real run's serves every draw."""
    prep = runs.prepare(book, forecaster)

    def chunk(seeds):
        return [runs.scored(runs.run(prep, 1.0, periods, seed=s), periods)["excess"] for s in seeds]

    parts = Parallel(n_jobs=jobs)(delayed(chunk)(s) for s in np.array_split(np.arange(draws), 20))
    return pd.DataFrame([x for part in parts for x in part], index=pd.RangeIndex(draws, name="draw"))


def placebo_shares(draws: pd.DataFrame, real: pd.DataFrame, months: pd.PeriodIndex) -> dict:
    """On these months: the share of shuffled spanning α's the real α beats (the placebo rule, pass at
    95%), and the share of shuffled Sharpe gains the real gain beats (information)."""
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
