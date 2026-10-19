"""
Build the Phase 7 results
=========================
    python -m src.walkforward.build              everything; the forecasts are cached, so a re-run takes minutes
    python -m src.walkforward.build --jobs 4     fewer parallel workers

Steps (development sample only):
  1. the real-time screen at every year-end and the recipes it gives
  2. every forecast every variant needs (8 models x 2 frequencies, one walk-forward per recipe), cached
  3. the 192 variants through the engine
  4. the primary test on its own, with its CER test, combination test and placebo
  5. the family on the months every variant scores: raw, Romano-Wolf and Holm p-values, the CER and
     combination tests, the deflated Sharpe ratio of the best variant
  6. Q3, every model against M0 cell by cell (168 tests, Romano-Wolf), and Q4, accuracy against value
  7. the ETF track and the two long-history references

Writes reports/phase7_screen.csv, phase7_recipes.csv, phase7_results.csv (the primary first, then every
variant), phase7_vs_m0.csv, phase7_q4.csv, phase7_tracks.csv and phase7_summary.csv; and in
data/processed/ the monthly returns of every run (phase7_runs.parquet), the primary run in full
(phase7_primary.parquet) and its placebo draws (phase7_placebo.parquet).
"""

from __future__ import annotations

import argparse
import time
from functools import reduce

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from src import engine as e
from src import models as md
from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data.load import load_dev, load_etf_ohlc_dev
from src.engine import evaluation as ev, execution
from src.models import scorecard as sc
from src.walkforward import FAMILY, FREQS, PRIMARY, RULES, runs, screened, stats, tracks

SLOW = ["M7", "M3", "M2", "M6", "M5", "M4"]          # walk-forwards worth caching, longest first
VARIANTS = [(m, r, x, f) for m in FAMILY["model"] for r in FAMILY["rule"] for x in FAMILY["mix"] for f in FAMILY["freq"]]
COLUMNS = ["model", "rule", "mix", "freq"]


def _index(variants: list[tuple]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(variants, names=COLUMNS)


# ---------------------------------------------------------------- 1-3: books, forecasts, runs

def setup(panel: pd.DataFrame, ohlc: pd.DataFrame, jobs: int):
    table = screened.screen_all(panel, ohlc, jobs)
    assets = screened.survivors(table)
    rec = screened.recipes(panel, assets)
    books = {(x, f): screened.book(panel, rec, x, FREQS[f]) for x in FAMILY["mix"] for f in FAMILY["freq"]}
    return table, assets, rec, books


def distinct_mixes(books: dict) -> list[pd.Series]:
    """The daily return of every distinct mix held (weekly books hold the same recipes as monthly ones)."""
    out = {}
    for (x, f), book in books.items():
        ids = e.data.recipe_ids(book.positions)
        for k in ids.unique():
            pos = book.positions.loc[ids == k].iloc[0]
            out.setdefault(tuple(pos[pos != 0].round(15).items()), e.data.mix(book, pos))
    return list(out.values())


def fill_cache(panel: pd.DataFrame, books: dict, jobs: int) -> None:
    mixes = distinct_mixes(books)
    for f in FREQS:                                  # monthly first: weekly GARCH reads the monthly fits
        fs = md.forecasters(panel, False, FREQS[f])
        t0 = time.time()
        Parallel(n_jobs=jobs, verbose=5)(delayed(fs[m])(p) for m in SLOW for p in mixes)
        print(f"{f} forecasts for {len(mixes)} mixes cached in {time.time() - t0:.0f}s", flush=True)


def cell(panel: pd.DataFrame, book: e.Book, freq: str, model: str) -> dict[str, pd.DataFrame]:
    """The three timing rules for one model, mix and frequency (the inverse rules share one forecast)."""
    f = md.forecasters(panel, False, FREQS[freq])[model]
    periods = execution.holding_periods(book)
    prep = runs.prepare(book, f)
    out = {r: runs.run(prep, RULES[r][0], periods) for r in ("inv_var", "inv_vol")}
    out["down_vol"] = runs.run(runs.prepare(book, runs.downside(f, FREQS[freq])), RULES["down_vol"][0], periods)
    return out


def family_runs(panel: pd.DataFrame, books: dict, jobs: int) -> dict[tuple, pd.DataFrame]:
    tasks = [(m, x, f) for m in FAMILY["model"] for x in FAMILY["mix"] for f in FAMILY["freq"]]
    done = Parallel(n_jobs=jobs, verbose=5)(delayed(cell)(panel, books[(x, f)], f, m) for m, x, f in tasks)
    return {(m, r, x, f): out[r] for (m, x, f), out in zip(tasks, done) for r in FAMILY["rule"]}


# ---------------------------------------------------------------- 4: the primary test

def single_tests(x: pd.Series, b: pd.Series) -> dict:
    """The CER test and the combination test for one run (bootstrap p: the larger over the two blocks)."""
    X = np.column_stack([x, b])
    cer_gain = ev.cer(x, runs.GAMMA) - ev.cer(b, runs.GAMMA)
    cer_p = max(stats.bootstrap_p(np.array([cer_gain]), _cer_gap(stats.counts(len(x), blk), X))[0]
                for blk in stats.BOOT["blocks"])
    out = {"cer_p": cer_p}
    comb = stats.combination(x, b)
    if len(comb) < 2:                                # under 120 training months: no out-of-sample month
        return {**out, "comb_months": len(comb), "comb_cer": np.nan, "comb_alone_cer": np.nan, "comb_gain": np.nan, "comb_p": np.nan}
    return {**out, **combination_row(comb)}


def _cer_gap(C: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Resampled CER(first column) - CER(second column), for column pairs (0, 1), (2, 3), ..."""
    c = stats.cer_draws(C, X, runs.GAMMA)
    return c[:, 0::2] - c[:, 1::2]


def combination_row(comb: pd.DataFrame) -> dict:
    X = comb[["combined", "alone"]].to_numpy()
    gain = ev.cer(comb["combined"], runs.GAMMA) - ev.cer(comb["alone"], runs.GAMMA)
    p = max(stats.bootstrap_p(np.array([gain]), _cer_gap(stats.counts(len(X), blk), X))[0] for blk in stats.BOOT["blocks"])
    return {"comb_months": len(comb), "comb_first": str(comb.index[0]), "comb_cer": ev.cer(comb["combined"], runs.GAMMA),
            "comb_alone_cer": ev.cer(comb["alone"], runs.GAMMA), "comb_gain": gain, "comb_p": p,
            "comb_w_managed": comb["w_managed"].mean(), "comb_w_bench": comb["w_bench"].mean(),
            "comb_w_alone": comb["w_alone"].mean()}


def placebo(book: e.Book, forecaster, periods, draws: int = e.ENGINE_RULES["placebo_draws"], jobs: int = -2) -> dict:
    """The engine's placebo on the walk-forward: each draw shuffles every recipe's forecast in time and
    reruns sizes (c_t included) and trading. The real spanning alpha must beat 95% of the shuffled ones."""
    prep = runs.prepare(book, forecaster)
    real = runs.scored(runs.run(prep, 1.0, periods), periods)

    def chunk(seeds):
        out = []
        for s in seeds:
            m = runs.scored(runs.run(prep, 1.0, periods, seed=s), periods)
            out.append(ev.alpha(m["excess"], m["bench_excess"]))
        return out

    seeds = np.array_split(np.arange(draws), 20)
    shuffled = np.concatenate(Parallel(n_jobs=jobs)(delayed(chunk)(s) for s in seeds))
    alpha = ev.alpha(real["excess"], real["bench_excess"])
    share = ev.placebo_share(alpha, shuffled)
    return {"alpha": alpha, "shuffled": shuffled, "share_beaten": share, "pass": bool(share >= e.ENGINE_RULES["placebo_share"])}


# ---------------------------------------------------------------- 5-6: the family

def family_table(R: dict, P: dict, months: pd.PeriodIndex) -> tuple[pd.DataFrame, dict]:
    """Every variant on the common months: metrics, the raw Ledoit-Wolf p, Romano-Wolf and Holm adjusted
    p-values for the Sharpe gain over buy-and-hold, and the CER and combination tests."""
    T = pd.DataFrame([runs.metrics(R[v], P[v[2], v[3]], months) for v in VARIANTS], index=_index(VARIANTS))
    X = {v: runs.scored(R[v], P[v[2], v[3]], months) for v in VARIANTS}
    xt = np.column_stack([X[v]["excess"] for v in VARIANTS])
    xb = np.column_stack([X[v]["bench_excess"] for v in VARIANTS])
    gain, se = T["gain"].astype(float).to_numpy(), T["se"].astype(float).to_numpy()
    cer_gain = T["cer_gain"].astype(float).to_numpy()
    rw, cer_p, draws = [], [], {}
    for blk in stats.BOOT["blocks"]:
        C = stats.counts(len(months), blk)
        st, sb = stats.sharpe_draws(C, xt), stats.sharpe_draws(C, xb)
        draws[blk] = st                                      # kept for Q3 (each model against M0)
        rw.append(stats.romano_wolf(gain / se, (st - sb - gain) / se))
        cer_p.append(stats.bootstrap_p(cer_gain, stats.cer_draws(C, xt, runs.GAMMA) - stats.cer_draws(C, xb, runs.GAMMA)))
    T["p_rw"], T["cer_p"], T["p_holm"] = np.max(rw, axis=0), np.max(cer_p, axis=0), stats.holm(T["p"].astype(float))
    T["edge_rw"], T["edge_holm"] = T["p_rw"] < 0.05, T["p_holm"] < 0.05

    combos = {v: stats.combination(X[v]["excess"], X[v]["bench_excess"]) for v in VARIANTS}
    xc = np.column_stack([combos[v][c] for v in VARIANTS for c in ("combined", "alone")])
    comb_gain = np.array([ev.cer(combos[v]["combined"], runs.GAMMA) - ev.cer(combos[v]["alone"], runs.GAMMA) for v in VARIANTS])
    comb_p = np.max([stats.bootstrap_p(comb_gain, _cer_gap(stats.counts(len(xc), blk), xc)) for blk in stats.BOOT["blocks"]], axis=0)
    T["comb_gain"], T["comb_p"] = comb_gain, comb_p
    T["comb_w_managed"] = [combos[v]["w_managed"].mean() for v in VARIANTS]
    T["comb_months"] = len(xc)
    return T, {"draws": draws, "X": X}


def vs_m0(T: pd.DataFrame, X: dict, draws: dict) -> pd.DataFrame:
    """Q3: in each of the 24 cells (rule x mix x frequency), Sharpe(model) - Sharpe(M0), with the
    Ledoit-Wolf p (one-sided: the model beats M0) and Romano-Wolf adjusted over the 168 comparisons."""
    pairs = [(v, ("M0",) + v[1:]) for v in VARIANTS if v[0] != "M0"]
    col = {v: i for i, v in enumerate(VARIANTS)}
    rows = []
    for v, base in pairs:
        d, se = ev.sharpe_difference(X[v]["excess"], X[base]["excess"])
        rows.append({"sharpe": T.loc[v, "sharpe"], "m0_sharpe": T.loc[base, "sharpe"], "diff": d, "se": se, "p": norm.sf(d / se)})
    Q = pd.DataFrame(rows, index=_index([v for v, _ in pairs]))
    d, se = Q["diff"].astype(float).to_numpy(), Q["se"].astype(float).to_numpy()
    rw = [stats.romano_wolf(d / se, (draws[blk][:, [col[v] for v, _ in pairs]] - draws[blk][:, [col[b] for _, b in pairs]] - d) / se)
          for blk in stats.BOOT["blocks"]]
    Q["p_rw"] = np.max(rw, axis=0)
    Q["beats_m0_rw"] = Q["p_rw"] < 0.05
    return Q


def q4(R: dict, books: dict, T: pd.DataFrame, Q: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Q4: Scorecard A (QLIKE, MCS) on the primary cell of the screened book, beside each model's value."""
    cell_runs = {m: R[(m,) + PRIMARY[1:]] for m in FAMILY["model"]}
    a = sc.scorecard_a(cell_runs, sc.targets(books[(PRIMARY[2], PRIMARY[3])]))
    gains = Q.groupby(level="model")["diff"].median()
    out = a[["qlike", "qlike_median", "rv_over_forecast", "mcs_p", "in_mcs", "qlike_rank", "months"]].copy()
    out["primary_cell_sharpe"] = [T.loc[(m,) + PRIMARY[1:], "sharpe"] for m in out.index]
    out["median_gain_over_m0"] = gains.reindex(out.index).fillna(0.0)
    out["cells_beating_m0"] = Q.groupby(level="model")["diff"].apply(lambda s: int((s > 0).sum())).reindex(out.index)
    out["variants_with_edge_rw"] = T.groupby(level="model")["edge_rw"].sum().reindex(out.index)
    rho = {"q4_rank_corr_primary_cell": float(out["qlike"].rank().corr(out["primary_cell_sharpe"].astype(float).rank(ascending=False), method="spearman")),
           "q4_rank_corr_median_gain": float(out.loc[out.index != "M0", "qlike"].rank().corr(
               out.loc[out.index != "M0", "median_gain_over_m0"].rank(ascending=False), method="spearman"))}
    return out, rho


# ---------------------------------------------------------------- 7: the other books

def track_rows(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, series = [], {}
    for name, s in tracks.specs(panel).items():
        periods = execution.holding_periods(s["book"], s["costs"])
        res = runs.run(runs.prepare(s["book"], s["forecaster"]), 1.0, periods, min_pairs=s["min_pairs"])
        m = runs.scored(res, periods)
        rows.append({"book": name, "label": s["label"], **runs.metrics(res, periods), **single_tests(m["excess"], m["bench_excess"])})
        series[name] = m
    return pd.DataFrame(rows).set_index("book"), series


# ---------------------------------------------------------------- main

def long_table(frames: dict, keys: list[str]) -> pd.DataFrame:
    parts = []
    for k, m in frames.items():
        k = k if isinstance(k, tuple) else (k,)
        parts.append(m.reset_index(names="month").assign(**dict(zip(keys, k))))
    out = pd.concat(parts, ignore_index=True)
    out["month"] = out["month"].astype(str)
    return out


def main(jobs: int = -2) -> None:
    t0 = time.time()
    panel, ohlc = load_dev(), load_etf_ohlc_dev()
    table, assets, rec, books = setup(panel, ohlc, jobs)
    table.to_csv(REPORTS_DIR / "phase7_screen.csv")
    rec.to_csv(REPORTS_DIR / "phase7_recipes.csv")
    print(f"screen and recipes: {len(rec.index.get_level_values('asof').unique())} year-ends hold something "
          f"({time.time() - t0:.0f}s)", flush=True)

    fill_cache(panel, books, jobs)
    R = family_runs(panel, books, jobs)
    P = {k: execution.holding_periods(b) for k, b in books.items()}
    print(f"192 variants run ({time.time() - t0:.0f}s)", flush=True)

    # 4. the primary test, on its own months
    res, per = R[PRIMARY], P[PRIMARY[2], PRIMARY[3]]
    m = runs.scored(res, per)
    pl = placebo(books[PRIMARY[2], PRIMARY[3]], md.forecasters(panel, False)[PRIMARY[0]], per, jobs=jobs)
    primary = {**runs.metrics(res, per), **single_tests(m["excess"], m["bench_excess"]),
               "placebo_share": pl["share_beaten"], "placebo_pass": pl["pass"]}
    print(f"primary: gain {primary['gain']:+.3f}, p {primary['p']:.4f}, {primary['reading']} ({time.time() - t0:.0f}s)", flush=True)

    # 5-6. the family on its common months
    months = reduce(lambda a, b: a.intersection(b), [runs.scored(R[v], P[v[2], v[3]]).index for v in VARIANTS])
    T, extra = family_table(R, P, months)
    Q = vs_m0(T, extra["X"], extra["draws"])
    A, rho = q4(R, books, T, Q)
    best = T["gain"].astype(float).idxmax()
    dsr = stats.deflated_sharpe(extra["X"][best]["excess"], T.loc[best, "bh_sharpe"] / np.sqrt(12),
                                float((T["gain"].astype(float) / np.sqrt(12)).var()), len(T))
    print(f"family: {int(T['edge_rw'].sum())} of {len(T)} variants beat buy-and-hold after Romano-Wolf, "
          f"{int(T['edge_holm'].sum())} after Holm; best {best}, DSR {dsr['dsr']:.3f} ({time.time() - t0:.0f}s)", flush=True)

    # 7. the other books
    TR, track_series = track_rows(panel)

    # write
    head = pd.DataFrame([{**dict(zip(COLUMNS, PRIMARY)), "row": "primary test (stands alone, its own months)", **primary}])
    body = T.reset_index().assign(row="variant (family, common months)")
    body["primary"] = [tuple(r) == PRIMARY for r in body[COLUMNS].to_numpy()]
    pd.concat([head.assign(primary=True), body], ignore_index=True).to_csv(REPORTS_DIR / "phase7_results.csv", index=False)
    Q.to_csv(REPORTS_DIR / "phase7_vs_m0.csv")
    A.to_csv(REPORTS_DIR / "phase7_q4.csv")
    TR.to_csv(REPORTS_DIR / "phase7_tracks.csv")
    summary = {"common_months": len(months), "common_first": str(months[0]), "common_last": str(months[-1]),
               "variants": len(T), "edge_raw": int((T["p"].astype(float) < 0.05).sum()), "edge_rw": int(T["edge_rw"].sum()),
               "edge_holm": int(T["edge_holm"].sum()), "q3_beats_m0_rw": int(Q["beats_m0_rw"].sum()), "q3_tests": len(Q),
               "best_variant": " | ".join(best), **{f"dsr_{k}": v for k, v in dsr.items()}, **rho,
               "placebo_alpha": pl["alpha"], "placebo_share": pl["share_beaten"]}
    pd.Series(summary).to_csv(REPORTS_DIR / "phase7_summary.csv", header=["value"])
    every = {v: runs.scored(R[v], P[v[2], v[3]]) for v in VARIANTS}
    long_table(every, COLUMNS).assign(book="family").pipe(
        lambda d: pd.concat([d, long_table(track_series, ["book"])], ignore_index=True)).to_parquet(PROCESSED_DIR / "phase7_runs.parquet")
    res.reset_index(names="month").assign(month=lambda d: d["month"].astype(str)).to_parquet(PROCESSED_DIR / "phase7_primary.parquet")
    pd.DataFrame({"alpha_shuffled": pl["shuffled"]}).to_parquet(PROCESSED_DIR / "phase7_placebo.parquet")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=-2, help="parallel workers (joblib n_jobs; default all but one)")
    main(ap.parse_args().jobs)
