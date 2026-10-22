"""
Build the Phase 9 results
=========================
    python -m src.holdout.build               the final test: opens the sealed holdout once, runs everything, writes
                                              reports/phase9_*.csv
    python -m src.holdout.build --rehearse    the same code on development data only, with a pseudo-holdout from
                                              2011 (REHEARSAL); writes data/processed/phase9_rehearsal/; never opens
                                              the holdout
    --jobs 4                                  fewer parallel workers

The final test runs once: it refuses to start, before opening anything, once reports/phase9_results.csv exists.

Steps:
  1. the panel: development + holdout. The paper books end with the last month every factor is complete,
     the ETF track with the last month SPY and MTUM are
  2. the real-time screen at every December year-end and its recipes; the survivors 2018-2025
  3. every book, its forecasts (cached, in parallel) and its run: the development walk-forward continued
  4. checks: each run's development months against its Phase 7 run, and the screen and recipes against Phase 7's
  5. the one-shot numbers on the holdout months: each book's test under (c) and (a), its metrics, the CER and
     combination tests, Lo's (2002) standard errors (information); and each holding year
  6. the primary's placebo (information)
  7. the 2020 and 2022 crises: daily paths, the three questions, the cap and costs
  8. the reading: each book's development and holdout outcome, and the plan's verdict

Writes (final test) reports/phase9_screen.csv, phase9_survivors.csv, phase9_checks.csv, phase9_results.csv,
phase9_years.csv, phase9_placebo.csv, phase9_crises.csv, phase9_verdict.csv and phase9_summary.csv; and in
data/processed/ every book's monthly returns (phase9_runs.parquet), the primary's run in full
(phase9_primary.parquet), the crisis windows day by day (phase9_paths.parquet) and the placebo's draws
(phase9_placebo_draws.parquet).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import engine as e
from src import models as md
from src import screen as sc
from src.config import FACTORS, HOLDOUT_START, PROCESSED_DIR, REPORTS_DIR
from src.data.load import HOLDOUT_PASSPHRASE, load_dev, load_etf_ohlc_dev, load_holdout
from src.engine import evaluation as ev
from src.holdout import (DEVELOPMENT, EPISODES, HOLDOUT_OUTCOME, HOLDOUT_RULES, NAMES, REHEARSAL, VERDICT, setup)
from src.robustness import checks, crises
from src.walkforward import build as wfb, runs, screened, stats

BOOKS = ["primary", "best", "spy", "etf", "fixed"]
PAPER = ["primary", "best", "fixed"]


# ---------------------------------------------------------------- 1: the panel

def panels(rehearse: bool, confirm: str = HOLDOUT_PASSPHRASE) -> dict:
    """The joined panel, the paper and ETF panels cut where their data end, the first holdout day, and the
    episodes. The rehearsal uses development data only, BAB losing its last month as AQR's file does.
    confirm: the loader's passphrase (Phase 10's runner, src/retest, passes its own)."""
    if rehearse:
        panel, ohlc = load_dev().copy(), load_etf_ohlc_dev()
        last = panel.index.to_period("M") == panel.index[-1].to_period("M")
        panel.loc[last, REHEARSAL["drop_last_month"]] = np.nan
        start, episodes = pd.Timestamp(REHEARSAL["start"]), REHEARSAL["episodes"]
    else:
        panel = setup.join(load_dev(), load_holdout(confirm=confirm))
        ohlc = setup.join(load_etf_ohlc_dev(), load_holdout(confirm=confirm, table="etf_ohlc"), date="date")
        start, episodes = HOLDOUT_START, EPISODES
    if start != pd.Timestamp(HOLDOUT_RULES["start"]) and not rehearse:
        raise ValueError("config.HOLDOUT_START and HOLDOUT_RULES['start'] disagree")
    paper_end = setup.complete_end(panel, FACTORS)
    etf_end = setup.complete_end(panel, ["spy", "mtum"])
    paper, paper_ohlc = setup.cut(panel, ohlc, paper_end)
    etf_panel, _ = setup.cut(panel, ohlc, etf_end)
    return {"panel": panel, "paper": paper, "ohlc": paper_ohlc, "etf_panel": etf_panel, "start": start,
            "paper_end": paper_end, "etf_end": etf_end, "episodes": episodes}


# ---------------------------------------------------------------- 3: forecasts and runs

def fill_cache(S: dict, paper: pd.DataFrame, jobs: int) -> None:
    """M3's monthly walk-forward on every mix the paper books hold and on momentum alone (the crisis runs),
    in parallel. The weekly M0 and the ETF track's GARCH need no cache."""
    t0 = time.time()
    mixes = wfb.distinct_mixes({(k, None): S[k]["book"] for k in ("primary", "fixed")})
    mixes.append(paper["mom"].dropna().rename("p"))
    m3 = md.frozen(True)
    Parallel(n_jobs=jobs, verbose=2)(delayed(m3)(p) for p in mixes)
    print(f"M3 on {len(mixes)} mixes ({time.time() - t0:.0f}s)", flush=True)


def all_runs(S: dict, jobs: int) -> tuple[dict, dict]:
    """Every book's prepared segments (forecasts; the ETF track's are computed here) and its run as registered."""
    preps = dict(zip(BOOKS, Parallel(n_jobs=jobs)(delayed(setup.prepare)(S[k]) for k in BOOKS)))
    return preps, {k: setup.run(S[k], preps[k]) for k in BOOKS}


# ---------------------------------------------------------------- 4: checks

def reproduction(R: dict) -> pd.DataFrame:
    """Each run against its Phase 7 run (data/processed/phase7_*.parquet), on every month both have: appending
    the holdout must change nothing before it. The weekly best variant is compared under Phase 7's scoring
    (each week in the month of its closing trade), except Phase 7's last month: the development data cut
    its last week short at 31 December 2018."""
    p7 = pd.read_parquet(PROCESSED_DIR / "phase7_runs.parquet")
    p7["month"] = pd.PeriodIndex(p7["month"], freq="M")
    model, rule, mix, freq = HOLDOUT_RULES["best"]
    refs = {"best": p7.query("book == 'family' and model == @model and rule == @rule and mix == @mix and freq == @freq"),
            "spy": p7.query("book == 'spy'"), "etf": p7.query("book == 'etf'")}
    rows = []
    for k, ref in refs.items():
        ref = ref.set_index("month")[["excess", "bench_excess", "size"]]
        if k == "best":
            ref = ref.iloc[:-1]
        new = runs.scored(R[k][1], R[k][2])
        rows.append({"book": k, "what": "monthly excess, bench_excess and size, against Phase 7", **compare(new, ref)})
    ref = pd.read_parquet(PROCESSED_DIR / "phase7_primary.parquet")
    ref.index = pd.PeriodIndex(ref.pop("month"), freq="M")
    res = R["primary"][1]
    cols = [c for c in ref.columns if c in res.columns and pd.api.types.is_numeric_dtype(ref[c])]
    rows.insert(0, {"book": "primary", "what": f"every numeric column of the run ({len(cols)}), against Phase 7",
                    **compare(res[cols].astype(float), ref[cols].astype(float))})
    return pd.DataFrame(rows)


def compare(new: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """The largest gap on the months both have, the months compared, and the reference's months up to the
    new run's last that the new run lacks."""
    m = ref.index.intersection(new.index)
    d = (new.loc[m] - ref.loc[m]).abs().to_numpy()
    return {"months": len(m), "first": str(m[0]) if len(m) else "", "last": str(m[-1]) if len(m) else "",
            "max_abs_diff": float(np.nanmax(d)) if len(m) else np.nan,
            "missing_months": len(ref.index[ref.index <= new.index[-1]].difference(new.index))}


def screen_check(table: pd.DataFrame, rec: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """The screen's survivors and the recipes' weights at every year-end before `start`, against Phase 7's
    tables: the number of (year-end, factor) verdicts that differ, and the largest gap in a weight (CSV)."""
    t7 = pd.read_csv(REPORTS_DIR / "phase7_screen.csv", parse_dates=["asof"]).set_index(["asof", "factor"])
    r7 = pd.read_csv(REPORTS_DIR / "phase7_recipes.csv", parse_dates=["asof"]).set_index(["asof", "method"])
    before = lambda df: df[df.index.get_level_values("asof") < start]
    held = lambda df: df[screened.FACTOR_FILTERS].all(axis=1) & (df["months"] >= sc.SCREEN_RULES["min_months"])
    a, b = held(before(table)), held(before(t7))
    w = [c for c in r7.columns if c.startswith("w_")]
    ra, rb = before(rec), before(r7)
    common = ra.index.intersection(rb.index)
    return pd.DataFrame([
        {"book": "screen", "what": "factors passing filters 1, 3 and 6 with 120 months, at every year-end before the holdout",
         "months": len(a), "max_abs_diff": float((a != b.reindex(a.index)).sum()), "missing_months": len(b.index.difference(a.index))},
        {"book": "recipes", "what": "mix weights at every year-end before the holdout (CSV precision)", "months": len(common),
         "max_abs_diff": float((ra.loc[common, w] - rb.loc[common, w]).abs().to_numpy().max()),
         "missing_months": len(rb.index.difference(ra.index))},
    ])


# ---------------------------------------------------------------- 5: the one-shot numbers

def holdout_months(R: dict, start: pd.Timestamp) -> dict[str, pd.PeriodIndex]:
    """The months each run scores in the holdout: held on every trading day (the ledger's whole months)."""
    first = pd.Period(start, "M")
    return {k: (lambda w: w[w >= first])(checks.whole_months(b, res)) for k, (b, res, _) in R.items()}


def results(R: dict, S: dict, months: dict, paper: pd.DataFrame, rec: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """One row per book and statistic: (c) for the screened books (each holding year scaled to an equal risk
    budget) and (a) for every book, with the metrics of Entry 6 (unscaled), the CER test, the combination
    test trained on every earlier scored month, and Lo's standard errors. Returns the table, each book's
    monthly returns (all scored months) and the (c) scales."""
    rows, series, scales = [], {}, {}
    for k in BOOKS:
        book, res, periods = R[k]
        m_all = checks.scored(book, res, periods)
        series[k] = m_all
        h = months[k]
        m = m_all.loc[h]
        row = checks.row(book, res, periods, h)                 # Entry 6's metrics, unscaled; the tests are below
        row = {c: v for c, v in row.items() if c not in ("gain", "se", "z", "p", "lower", "upper", "mde", "reading")}
        cer_p = wfb.single_tests(m["excess"], m["bench_excess"])["cer_p"]
        comb = stats.combination(m_all["excess"], m_all["bench_excess"])
        comb = comb[comb.index.isin(h)]
        comb_row = wfb.combination_row(comb) if len(comb) >= 2 else {"comb_months": len(comb)}
        lo = {"lo_se": setup.lo_se(m["excess"]), "bh_lo_se": setup.lo_se(m["bench_excess"])}
        base = {"book": k, "name": NAMES[k], **row, "cer_p": cer_p, **comb_row, **lo}
        stat = {"(a)": m}
        if S[k]["mix"] is not None:
            scales[k] = screened.year_scale(paper, rec, S[k]["mix"])
            stat = {"(c)": runs.year_weighted(m, scales[k]), "(a)": m}
        for name, x in stat.items():
            t = ev.sharpe_gain_test(x)
            rows.append({**base, "statistic": name, "decides": HOLDOUT_RULES["decides"][k] == name,
                         "stat_sharpe": ev.sharpe(x["excess"]), "stat_bh_sharpe": ev.sharpe(x["bench_excess"]),
                         **{f"stat_{c}": t[c] for c in ("gain", "se", "z", "p", "lower", "upper", "mde", "reading")}})
    return pd.DataFrame(rows), series, scales


def years(series: dict, months: dict, scales: dict, rec: pd.DataFrame, S: dict) -> pd.DataFrame:
    """Each book's holding years in the holdout: the assets held, months, compounded excess returns of both
    runs, realized volatilities, the mean size, and σ̂_y with its scale for the screened books."""
    rows = []
    for k in BOOKS:
        m = series[k].loc[months[k]]
        for y, g in m.groupby(m.index.year):
            row = {"book": k, "year": y, "months": len(g),
                   "timed": float(np.prod(1 + g["excess"]) - 1), "bh": float(np.prod(1 + g["bench_excess"]) - 1),
                   "timed_vol": g["excess"].std() * np.sqrt(12), "bh_vol": g["bench_excess"].std() * np.sqrt(12),
                   "mean_size": g["size"].mean()}
            if k in scales:
                row |= {"sigma_hat": 1 / scales[k].get(y, np.nan), "scale": scales[k].get(y, np.nan)}
                held = rec.xs(S[k]["mix"], level="method")["assets"]
                row["assets"] = held[held.index.year == y - 1].iloc[0] if (held.index.year == y - 1).any() else ""
            rows.append(row)
    return pd.DataFrame(rows)


def survivors_table(table: pd.DataFrame, rec: pd.DataFrame, paper: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    """Entry 10's report: the survivors at each year-end from the last before the holdout, with the duplicates
    dropped, the inverse-volatility and shrunk mean-variance weights and each mix's σ̂."""
    ends = [d for d in table.index.get_level_values("asof").unique() if d >= start - pd.DateOffset(years=1)]
    rows = []
    for d in ends:
        t = table.loc[d]
        ok = t[["f1", "f3", "f6"]].all(axis=1) & (t["months"] >= 120)
        row = {"asof": d.date(), "held_in": d.year + 1, "survivors": " ".join(t.index[ok])}
        for method in ("inv_vol", "shrunk_mv"):
            if (d, method) in rec.index:
                r = rec.loc[(d, method)]
                row |= {f"{method}_{c[2:]}": r[c] for c in rec.columns if c.startswith("w_") and r[c] != 0}
                row["dropped"] = r["dropped"]
                row[f"{method}_sigma_hat"] = 1 / screened.year_scale(paper, rec.loc[[(d, method)]], method).iloc[0]
        for f in t.index:
            row[f"{f}_phi"], row[f"{f}_t_alpha"] = t.loc[f, "phi"], t.loc[f, "t_alpha"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 6: the placebo

def placebo(R: dict, S: dict, months: pd.PeriodIndex, jobs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Phase 7's placebo on the primary (1,000 shuffled forecasts, seeds 0-999), read on its holdout months."""
    book, res, periods = R["primary"]
    draws = wfb.placebo(book, S["primary"]["forecaster"], periods, HOLDOUT_RULES["placebo_draws"], jobs)["draws"]
    real = runs.scored(res, periods)
    return pd.DataFrame([{"book": "primary", **checks.placebo_shares(draws, real, months)}]), draws


# ---------------------------------------------------------------- 7: the crises

def crisis_runs(paper: pd.DataFrame, S: dict, preps: dict, R: dict) -> dict:
    """(book, version) -> {variation: (book, res, periods)}: each run as registered, with no cap, with a cap
    of 1.0 and at 30 bp a trade (the ETF track keeps its own costs), and the single factors of Entry 8."""
    k30 = 30e-4
    out = {}
    for k in ("primary", "best", "fixed", "spy", "etf"):
        v = {"as run": R[k], "no cap": setup.run(S[k], preps[k], max_size=None), "cap 1.0": setup.run(S[k], preps[k], max_size=1.0)}
        if k in PAPER:
            v["30 bp"] = setup.run(S[k], preps[k], costs=setup.kappa_everywhere(S[k]["book"], k30))
        out[(k, "frozen")] = v
    for factor in ("mom", "mkt"):
        cost = e.Costs(kappa={factor: k30})
        out[(factor, "bsc")] = {"as run": crises.managed(paper, factor, "bsc"), "cap 1.5": crises.managed(paper, factor, "bsc", 1.5),
                                "cap 1.0": crises.managed(paper, factor, "bsc", 1.0), "30 bp": crises.managed(paper, factor, "bsc", costs=cost)}
    cost = e.Costs(kappa={"mom": k30})
    out[("mom", "frozen")] = {"as run": crises.managed(paper, "mom", "frozen"), "no cap": crises.managed(paper, "mom", "frozen", None),
                              "cap 1.0": crises.managed(paper, "mom", "frozen", 1.0), "30 bp": crises.managed(paper, "mom", "frozen", costs=cost)}
    return out


# The forecast's horizon in trading days, to read σ̂ as an annual volatility: monthly GARCH 22, weekly M0 5.
HORIZON = {"primary": 22, "best": 5, "fixed": 22, "spy": 22, "etf": 22, "mom": 22}


def crisis_tables(paper: pd.DataFrame, C: dict, episodes: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry 8's three questions for every episode and every run that holds a position on every day of its
    window, and the windows day by day (from two months before, for the realized volatility's warm-up)."""
    rows, paths = [], []
    P = {k: crises.path(*v["as run"]) for k, v in C.items()}
    monthly = {k: {name: checks.scored(*r)["excess"] for name, r in v.items()} for k, v in C.items()}
    bench = {k: checks.scored(*v["as run"])["bench_excess"] for k, v in C.items()}
    for episode, (peak_from, trough_year) in episodes.items():
        s = setup.span(paper, peak_from, trough_year)
        for (book, version), variants in C.items():
            res = variants["as run"][1]
            p = P[(book, version)]
            if not crises.covers(p, s, paper.index):
                print(f"  {episode}: {book} ({version}) does not hold every day of the window; not shown", flush=True)
                continue
            q = crises.questions(p, s)
            rows.append({"episode": episode, "book": book, "version": version, "peak": s["peak"].date(), "trough": s["trough"].date(),
                         "window_start": s["start"].date(), "window_end": s["end"].date(), "market_fall": s["market_fall"], **q,
                         "window_bench": crises.window_growth(bench[(book, version)], s["months"]),
                         **{f"window: {name}": crises.window_growth(x, s["months"]) for name, x in monthly[(book, version)].items()},
                         "window_capped_share": res.loc[np.asarray(pd.PeriodIndex(res.index.start_time, freq="M").isin(s["months"])), "capped"].mean()})
            w = p.loc[s["start"] - pd.DateOffset(months=2):s["end"]]
            horizon = 21 if version == "bsc" else HORIZON[book]           # Barroso and Santa-Clara: 21 x mean r²
            paths.append(w.assign(episode=episode, book=book, version=version, horizon=horizon).reset_index(names="date"))
    return pd.DataFrame(rows), (pd.concat(paths, ignore_index=True) if paths else pd.DataFrame())


# ---------------------------------------------------------------- 8: the reading

def verdict(T: pd.DataFrame) -> pd.DataFrame:
    """Each book: its development outcome (Entry 13, from Entries 5 and 7), its holdout reading under the
    statistic that decides it, the plan's column for that reading, and the plan's verdict."""
    rows = []
    for k in BOOKS:
        r = T[(T["book"] == k) & T["decides"]].iloc[0]
        dev, dev_note = DEVELOPMENT[k]
        outcome = HOLDOUT_OUTCOME[r["stat_reading"]]
        rows.append({"book": k, "name": NAMES[k], "development": dev, "development_result": dev_note,
                     "statistic": r["statistic"], "holdout_gain": r["stat_gain"], "holdout_p": r["stat_p"],
                     "holdout_reading": r["stat_reading"], "holdout": outcome, "verdict": VERDICT[(dev, outcome)],
                     "role": "information" if k == "fixed" else ("the final test" if k == "primary" else "secondary")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def write(df: pd.DataFrame, out: Path, name: str, **kw) -> None:
    path = out / name
    (df.to_parquet(path) if name.endswith(".parquet") else df.to_csv(path, **kw))


def main(jobs: int = -2, rehearse: bool = False) -> None:
    t0 = time.time()
    if not rehearse and (REPORTS_DIR / "phase9_results.csv").exists():
        raise SystemExit("the final test has already run (reports/phase9_results.csv exists); the holdout is opened once")
    state = {"mode": "rehearsal (development data only)" if rehearse else "final test"}
    reports = PROCESSED_DIR / "phase9_rehearsal" if rehearse else REPORTS_DIR
    data = reports if rehearse else PROCESSED_DIR
    reports.mkdir(parents=True, exist_ok=True)

    D = panels(rehearse)
    paper, start = D["paper"], D["start"]
    state |= {"panel_first": D["panel"].index[0].date(), "panel_last": D["panel"].index[-1].date(), "holdout_first": start.date(),
              "paper_end": D["paper_end"].date(), "etf_end": D["etf_end"].date()}
    print(f"panel {state['panel_first']} to {state['panel_last']}; paper books end {state['paper_end']}, "
          f"ETF track {state['etf_end']} ({time.time() - t0:.0f}s)", flush=True)

    table, surv = setup.screen(paper, D["ohlc"], jobs)
    rec = screened.recipes(paper, surv)
    write(table[table.index.get_level_values("asof") >= start - pd.DateOffset(years=1)], reports, "phase9_screen.csv")
    write(survivors_table(table, rec, paper, start), reports, "phase9_survivors.csv", index=False)
    print(f"screen: {len(surv)} year-ends; held after the last: {surv.iloc[-1]} ({time.time() - t0:.0f}s)", flush=True)

    S = setup.specs(paper, D["etf_panel"], rec, start)
    fill_cache(S, paper, jobs)
    preps, R = all_runs(S, jobs)
    months = holdout_months(R, start)
    state |= {f"months_{k}": len(m) for k, m in months.items()}
    state |= {f"first_{k}": str(m[0]) for k, m in months.items() if len(m)}
    state |= {f"last_{k}": str(m[-1]) for k, m in months.items() if len(m)}
    print("runs: " + ", ".join(f"{k} {len(m)} months" for k, m in months.items()) + f" ({time.time() - t0:.0f}s)", flush=True)

    K = pd.concat([reproduction(R), screen_check(table, rec, start)], ignore_index=True)
    write(K, reports, "phase9_checks.csv", index=False)
    print(K.to_string(), flush=True)

    T, series, scales = results(R, S, months, paper, rec)
    write(T, reports, "phase9_results.csv", index=False)
    write(years(series, months, scales, rec, S), reports, "phase9_years.csv", index=False)
    write(long_table(series, months), data, "phase9_runs.parquet")
    res = R["primary"][1]
    write(res.reset_index(names="month").assign(month=lambda d: d["month"].astype(str)), data, "phase9_primary.parquet")
    V = verdict(T)
    write(V, reports, "phase9_verdict.csv", index=False)
    print(T.loc[T["decides"], ["book", "statistic", "months", "stat_sharpe", "stat_bh_sharpe", "stat_gain", "stat_se", "stat_p",
                               "stat_reading"]].to_string(), flush=True)
    print(f"one-shot numbers written ({time.time() - t0:.0f}s)", flush=True)

    PL, draws = placebo(R, S, months["primary"], jobs)
    write(PL, reports, "phase9_placebo.csv", index=False)
    write(draws.loc[:, draws.columns.isin(months["primary"])].rename(columns=str), data, "phase9_placebo_draws.parquet")
    print(f"placebo: real alpha beats {PL['alpha_share'].iloc[0]:.1%} ({time.time() - t0:.0f}s)", flush=True)

    C = crisis_runs(paper, S, preps, R)
    Q, paths = crisis_tables(paper, C, D["episodes"])
    write(Q, reports, "phase9_crises.csv", index=False)
    write(paths, data, "phase9_paths.parquet")
    print(f"crises: {len(Q)} episode-runs ({time.time() - t0:.0f}s)", flush=True)

    state |= {"seconds": round(time.time() - t0), "lo_example": setup.lo_example()["se_annual"]}
    write(pd.Series(state), reports, "phase9_summary.csv", header=["value"])
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def long_table(series: dict, months: dict) -> pd.DataFrame:
    """Every book's monthly returns over all its scored months, with whether each month is in the holdout."""
    parts = [m.assign(book=k, holdout=m.index.isin(months[k])).reset_index(names="month") for k, m in series.items()]
    out = pd.concat(parts, ignore_index=True)
    out["month"] = out["month"].astype(str)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=-2, help="parallel workers (joblib n_jobs; default all but one)")
    ap.add_argument("--rehearse", action="store_true", help="development data only, pseudo-holdout from 2011; never opens the holdout")
    args = ap.parse_args()
    main(args.jobs, args.rehearse)
