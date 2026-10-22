"""
Build Track B's development results
===================================
    python -m src.fixes.build --mechanics --books twin spy   every variant on each book, end to end on development
                                                             data; prints counts only and writes nothing
    python -m src.fixes.build                                the development walk-forward; refuses until OPEN is
                                                             decided. Writes reports/phase10_trackb_dev_*.csv
    --jobs 4                                                 fewer parallel workers

Steps:
  1. the books on development data: the twin (the market factor at SPY's costs, 1946-2018), SPY alone (from 1997)
     and the paper mix (Phase 7's primary: the real-time screen, M3)
  2. each book's frozen run and the four variants, scored on the frozen run's months
  3. each book's family: the primary test of every variant ((c) for the paper mix, (a) otherwise), Romano-Wolf
     and Holm across the four. The primary book's development family decides (the twin's if SPY is primary)
  4. the development crises (Entry 8's episodes, B4's design): the three questions for every run covering one
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import engine as e
from src import models as md
from src.config import REPORTS_DIR
from src.data.load import load_dev, load_etf_ohlc_dev
from src.engine import evaluation as ev, execution
from src.engine.data import MONTH
from src.fixes import OPEN, TRACK_B, require_decided, rules
from src.holdout import setup as hsetup
from src.robustness import crises
from src.walkforward import PRIMARY, build as wfb, runs, screened, stats, tracks

BOOK_NAMES = ("twin", "spy", "paper")
DEV_FOR = {"spy": "twin", "paper": "paper"}            # the primary book -> the development family that decides


# ---------------------------------------------------------------- 1: the books

def specs(panel: pd.DataFrame, ohlc: pd.DataFrame, books: list[str], jobs: int) -> dict[str, dict]:
    """Each book: book, forecaster, costs, c_t minimum, and the (c) scale by holding year (screened books only)."""
    t = tracks.specs(panel)
    out = {}
    for k in books:
        if k in ("twin", "spy"):
            s = t["market" if k == "twin" else "spy"]
            out[k] = {"book": s["book"], "forecaster": s["forecaster"], "costs": s["costs"], "min_pairs": s["min_pairs"],
                      "scale": None}
        elif k == "paper":
            model, _, mix, _ = PRIMARY
            _, surv = hsetup.screen(panel, ohlc, jobs)
            rec = screened.recipes(panel, surv)
            book = screened.book(panel, rec, mix, MONTH)
            fill_cache(book, jobs)
            out[k] = {"book": book, "forecaster": md.forecasters(panel, False)[model], "costs": e.COSTS["paper"],
                      "min_pairs": runs.MIN_PAIRS[MONTH], "scale": screened.year_scale(panel, rec, mix)}
        else:
            raise ValueError(f"unknown book {k!r}: one of {BOOK_NAMES}")
    return out


def fill_cache(book: e.Book, jobs: int) -> None:
    """M3's walk-forward on every mix the screened book holds, in parallel (Phase 9's fill)."""
    t0 = time.time()
    mixes = wfb.distinct_mixes({("paper", None): book})
    m3 = md.frozen(True)
    Parallel(n_jobs=jobs, verbose=2)(delayed(m3)(p) for p in mixes)
    print(f"M3 on {len(mixes)} mixes ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 2: the runs

def book_runs(spec: dict) -> tuple[dict[str, pd.DataFrame], execution.Periods]:
    """The frozen run and the four variants, each at its own power (TRACK_B["power"]). B2 runs on the downside
    forecast, the rest on the book's own."""
    f = spec["forecaster"]
    periods = execution.holding_periods(spec["book"], spec["costs"])
    base = runs.prepare(spec["book"], f)
    down = runs.prepare(spec["book"], runs.downside(f, MONTH))
    R = {v: rules.run(down if v == "B2" else base, periods, v, min_pairs=spec["min_pairs"]) for v in rules.VARIANTS}
    return R, periods


def common_months(R: dict, periods: execution.Periods) -> pd.PeriodIndex:
    sets = [runs.scored(res, periods).index for res in R.values()]
    out = sets[0]
    for s in sets[1:]:
        out = out.intersection(s)
    return out


# ---------------------------------------------------------------- 3: the family

def family_tests(X: dict[str, pd.DataFrame], blocks=stats.BOOT["blocks"]) -> pd.DataFrame:
    """X: variant -> monthly excess and bench_excess on the same months (scaled already under (c)). The primary
    test of each, then Romano-Wolf (Phase 7's: the stationary bootstrap, the larger p over the blocks) and Holm
    across the family."""
    T = pd.DataFrame({v: ev.sharpe_gain_test(x) for v, x in X.items()}).T
    fam = list(X)
    gain, se = T["gain"].astype(float).to_numpy(), T["se"].astype(float).to_numpy()
    xt = np.column_stack([X[v]["excess"] for v in fam])
    xb = np.column_stack([X[v]["bench_excess"] for v in fam])
    rw = []
    for blk in blocks:
        C = stats.counts(len(xt), blk)
        rw.append(stats.romano_wolf(gain / se, (stats.sharpe_draws(C, xt) - stats.sharpe_draws(C, xb) - gain) / se))
    T["p_rw"], T["p_holm"] = np.max(rw, axis=0), stats.holm(T["p"].astype(float))
    T["survives_rw"], T["survives_holm"] = T["p_rw"] < 0.05, T["p_holm"] < 0.05
    return T.rename_axis("variant")


def book_table(name: str, spec: dict, R: dict, periods: execution.Periods, months: pd.PeriodIndex) -> pd.DataFrame:
    """Every run's metrics on the common months (unscaled), and the family's tests under the book's statistic."""
    M = pd.DataFrame({v: runs.metrics(R[v], periods, months) for v in rules.VARIANTS}).T
    scored = {v: runs.scored(R[v], periods, months) for v in TRACK_B["family"]}
    stat = "(a)"
    if spec["scale"] is not None:
        scored = {v: runs.year_weighted(m, spec["scale"]) for v, m in scored.items()}
        stat = "(c)"
    T = family_tests(scored).add_prefix("stat_")
    out = M.join(T, how="left").rename_axis("variant").reset_index()
    return out.assign(book=name, statistic=stat)


# ---------------------------------------------------------------- 4: the development crises

def crisis_table(panel: pd.DataFrame, name: str, spec: dict, R: dict, periods: execution.Periods) -> pd.DataFrame:
    rows = []
    paths = {v: crises.path(spec["book"], res, periods) for v, res in R.items()}
    for episode, (first, last) in crises.EPISODES.items():
        s = crises.span(panel, first, last)
        for v, p in paths.items():
            row = {"book": name, "episode": episode, "variant": v, "covered": crises.covers(p, s, panel.index)}
            if row["covered"]:
                row |= crises.questions(p, s)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main(mechanics: bool, books: list[str], jobs: int = -2) -> None:
    t0 = time.time()
    if not mechanics:
        require_decided()
        books = list(BOOK_NAMES)
    panel, ohlc = load_dev(), load_etf_ohlc_dev()
    S = specs(panel, ohlc, books, jobs)
    tables, crisis = [], []
    for name, spec in S.items():
        R, periods = book_runs(spec)
        months = common_months(R, periods)
        same = {v: runs.scored(res, periods).index.equals(months) for v, res in R.items()}
        T = book_table(name, spec, R, periods, months)
        C = crisis_table(panel, name, spec, R, periods)
        if mechanics:                                  # counts only: no return, size, test or crisis number
            fam = T[T["variant"].isin(TRACK_B["family"])].drop(columns=["book", "statistic", "variant"])
            print(f"{name}: {len(months)} common months, {months[0]} to {months[-1]}; every run on exactly those months: "
                  f"{all(same.values())}; table {T.shape[0]} rows x {T.shape[1]} columns, "
                  f"{int(fam.isna().to_numpy().sum())} empty cells in the family's rows; "
                  f"crisis windows covered {int(C['covered'].sum())} of {len(C)} ({time.time() - t0:.0f}s)", flush=True)
            continue
        dev = DEV_FOR[OPEN["primary_book"]]
        tables.append(T.assign(decides=name == dev))
        crisis.append(C)
    if mechanics:
        print("mechanics only: nothing written", flush=True)
        return
    pd.concat(tables, ignore_index=True).to_csv(REPORTS_DIR / "phase10_trackb_dev_family.csv", index=False)
    pd.concat(crisis, ignore_index=True).to_csv(REPORTS_DIR / "phase10_trackb_dev_crises.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mechanics", action="store_true", help="counts only: run end to end, print no result, write nothing")
    ap.add_argument("--books", nargs="+", default=list(BOOK_NAMES), choices=BOOK_NAMES, help="books for --mechanics")
    ap.add_argument("--jobs", type=int, default=-2, help="parallel workers (joblib n_jobs; default all but one)")
    args = ap.parse_args()
    main(args.mechanics, args.books, args.jobs)
