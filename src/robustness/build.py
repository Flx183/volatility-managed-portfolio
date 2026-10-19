"""
Build the Phase 8 results
=========================
    python -m src.robustness.build                 everything; forecasts are cached, so a re-run takes minutes
    python -m src.robustness.build --mechanics     books, forecasts and runs, then only dates and counts
    python -m src.robustness.build --jobs 4        fewer parallel workers

Steps (development sample only):
  1. the books: the real-time screen (the primary's) and Phase 4's fixed recipes, each also with one
     asset dropped at a time
  2. every forecast the checks need, cached: M3 on every new mix and on each asset alone, M3 refit on
     rolling windows, and M3 one week and one day ahead
  3. every one-at-a-time change through the engine, on both book families
  4. the robustness table: every change scored on its base's months, each era on its own; each change's
     effect on the gain split into the spec effect (same months, paired bootstrap interval) and the sample
     effect (months lost); and each check's verdict against its base
  5. the placebo: 1,000 shuffled forecasts for each book family, read on every base and era
  6. the crises: daily paths, the three questions, and momentum's two crashes
  7. information: each asset's share of buy-and-hold's risk, and the ETF track's short-fee stress
  8. information, after PREREGISTRATION Entry 10: the primary under option (c), every holding year
     scaled to an equal risk budget, beside the pre-registered (a)

Writes reports/phase8_robustness.csv, phase8_checks.csv, phase8_placebo.csv, phase8_crises.csv,
phase8_crashes.csv, phase8_risk_shares.csv, phase8_etf_stress.csv, phase8_year_weighted.csv and
phase8_year_scales.csv; and in data/processed/ the
monthly returns of every row (phase8_runs.parquet), the crisis windows day by day, from two months
before each window (phase8_paths.parquet), and the placebo's shuffled runs (phase8_placebo_draws.parquet).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import engine as e
from src import models as md
from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data.load import load_dev, load_etf_ohlc_dev
from src.engine import evaluation as ev, execution
from src.engine.data import DAY, MONTH, WEEK
from src.robustness import BASES, ROBUST_RULES, checks, crises
from src.robustness.checks import BASE, Spec
from src.walkforward import build as wfb, runs, screened, tracks

FAMILY = {"primary": "screened", "primary_1997": "screened", "fixed": "fixed"}
CUT_1997 = pd.period_range("1997-01", "2018-12", freq=MONTH)


def key(change: dict) -> Spec:
    """The run behind one of CHECKS' changes: its Spec without the era (an era rescores the base run)."""
    return checks.spec_of({k: v for k, v in change.items() if k != "era"})


# ---------------------------------------------------------------- 1-2: books and forecasts

def setup(panel: pd.DataFrame, ohlc: pd.DataFrame, jobs: int) -> dict:
    """Recipes of both families, with nothing or one asset dropped, and the assets each holds."""
    surv = screened.survivors(screened.screen_all(panel, ohlc, jobs))
    rec = {("screened", None): checks.recipes(panel, "screened", surv), ("fixed", None): checks.recipes(panel, "fixed")}
    assets = {fam: checks.held_assets(rec[(fam, None)]) for fam in ("screened", "fixed")}
    for fam in assets:
        for a in assets[fam]:
            rec[(fam, a)] = checks.recipes(panel, fam, surv, drop=a)
    return {"survivors": surv, "rec": rec, "assets": assets}


def fill_cache(panel: pd.DataFrame, rec: dict, assets: dict, jobs: int) -> None:
    """M3's monthly walk-forward on every mix and asset first (the weekly and daily forecasts read its
    fits), then the rolling windows, weekly and daily forecasts of the two base books' mixes."""
    t0 = time.time()
    every = wfb.distinct_mixes({k: checks.book(panel, r) for k, r in rec.items()})
    singles = [panel[a].dropna().rename("p") for a in sorted(set(assets["screened"]) | set(assets["fixed"]))]
    m3 = md.frozen(True)
    Parallel(n_jobs=jobs, verbose=2)(delayed(m3)(p) for p in every + singles)
    print(f"M3 monthly on {len(every)} mixes and {len(singles)} assets ({time.time() - t0:.0f}s)", flush=True)
    base = wfb.distinct_mixes({(k, None): checks.book(panel, rec[(k, None)]) for k in ("screened", "fixed")})
    fs = [md.frozen(True, window=d) for d, _ in ROBUST_RULES["windows"].values()] + [md.frozen(True, WEEK), md.frozen(True, DAY)]
    Parallel(n_jobs=jobs, verbose=2)(delayed(f)(p) for f in fs for p in base)
    print(f"rolling, weekly and daily M3 on {len(base)} mixes ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 3: the runs

def tasks(assets: dict) -> list[tuple[str, Spec]]:
    """Every distinct run: (family, spec). Eras reuse the base run; a drop changes the recipes."""
    out = {(fam, key(change)) for fam in ("screened", "fixed") for _, _, change in checks.variations(assets[fam])}
    return sorted(out, key=str)


def one(panel: pd.DataFrame, rec: dict, fam: str, spec: Spec):
    return checks.run(panel, rec[(fam, spec.drop)], spec)


def all_runs(panel: pd.DataFrame, rec: dict, assets: dict, jobs: int) -> dict:
    todo = tasks(assets)
    done = Parallel(n_jobs=jobs, verbose=2)(delayed(one)(panel, rec, fam, spec) for fam, spec in todo)
    return dict(zip(todo, done))


# ---------------------------------------------------------------- 4: the table

def base_months(R: dict) -> dict[str, pd.PeriodIndex]:
    out = {}
    for base, fam in FAMILY.items():
        b, res, _ = R[(fam, BASE)]
        m = checks.whole_months(b, res)
        out[base] = m.intersection(CUT_1997) if base == "primary_1997" else m
    return out


def era_months(months: pd.PeriodIndex, era: str) -> pd.PeriodIndex:
    lo, hi = ROBUST_RULES["eras"][era]
    return months[(months >= pd.Period(f"{lo}-01", MONTH)) & (months <= pd.Period(f"{hi}-12", MONTH))]


def table(R: dict, assets: dict, months: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One row per base, check and variation (the base's own setting included), and the monthly
    series behind each. A variation is scored on the base's months it also holds on every trading day;
    its effect on the gain is split into the spec effect (against the base on those same months) and the
    sample effect (the base on those months against the base on all of its months)."""
    rows, series = [], {}
    for base, fam in FAMILY.items():
        b0, res0, per0 = R[(fam, BASE)]
        base_all = checks.scored(b0, res0, per0, months[base])
        for check, label, change in checks.variations(assets[fam]):
            if base == "primary_1997" and check == "subperiod":
                continue                                        # 1997-2018 sits inside one era
            b, res, periods = R[(fam, key(change))]
            m = era_months(months[base], change["era"]) if "era" in change else months[base].intersection(checks.whole_months(b, res))
            r = checks.row(b, res, periods, m)
            x = checks.scored(b, res, periods, m)
            fx = checks.effects(base_all.loc[m], x, base_all) if len(m) >= 3 else {}
            rows.append({"base": base, "check": check, "variation": label, "is_base": not change, **r, **fx,
                         "months_lost": 0 if "era" in change else len(months[base]) - len(m)})
            if len(m):
                series[(base, check, label)] = x
    T = pd.DataFrame(rows)
    return T, verdicts(T), series


def verdicts(T: pd.DataFrame) -> pd.DataFrame:
    """Each check against its base: Entry 8's verdict on the gains and readings (a gain of exactly 0 is a
    tie, not a flip), and beside it the spec effects (none for eras, which change only the months): how
    many have a 95% interval wholly above or below 0."""
    rows = []
    for (base, check), g in T.groupby(["base", "check"], sort=False):
        b = T[(T["base"] == base) & T["is_base"] & (T["check"] == "frequency")].iloc[0]
        var = g[~g["is_base"] & g["reading"].isin(["edge", "inconclusive", "no meaningful edge"])]
        spec = var.iloc[:0] if check == "subperiod" else var
        rows.append({"base": base, "check": check, "base_gain": b["gain"], "base_reading": b["reading"],
                     "variations_read": len(var), "gains_positive": int((var["gain"] > 0).sum()),
                     "edge": int((var["reading"] == "edge").sum()), "min_gain": var["gain"].min(), "max_gain": var["gain"].max(),
                     "verdict": checks.verdict(b["gain"], b["reading"], list(var["gain"]), list(var["reading"])) if len(var) else "–",
                     "spec_min": spec["spec_effect"].min(), "spec_max": spec["spec_effect"].max(),
                     "spec_above_0": int((spec["spec_lower"] > 0).sum()), "spec_below_0": int((spec["spec_upper"] < 0).sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5: the placebo

def placebos(panel: pd.DataFrame, R: dict, months: dict, jobs: int) -> tuple[pd.DataFrame, dict]:
    """Phase 7's placebo (walkforward.build.placebo: inverse-variance timing, cap 1.5, 120 pairs: the base
    spec) on each base book, read on each base's months and eras. The books are monthly, so Phase 7's
    monthly returns are the ledger's."""
    rows, draws = [], {}
    for fam in ("screened", "fixed"):
        b, res, periods = R[(fam, BASE)]
        draws[fam] = wfb.placebo(b, checks.resolve(BASE)["forecaster"], periods, ROBUST_RULES["placebo_draws"], jobs)["draws"]
        real = runs.scored(res, periods)
        for base in [k for k, f in FAMILY.items() if f == fam]:
            sets = {"all months": months[base]} | {era: era_months(months[base], era) for era in ROBUST_RULES["eras"]}
            for name, m in sets.items():
                if len(m) >= ROBUST_RULES["min_era_months"] or name == "all months":
                    rows.append({"base": base, "months_set": name, **checks.placebo_shares(draws[fam], real, m)})
    return pd.DataFrame(rows), draws


# ---------------------------------------------------------------- 6: the crises

def crisis_runs(panel: pd.DataFrame, R: dict) -> dict:
    """(book, version) -> {"as run", "no cap", "cap 1.0", "cap 1.5", "30 bp"}: (book, res, periods)."""
    k30 = {"kappa": 30e-4}
    out = {}
    for name, fam in (("primary", "screened"), ("fixed", "fixed")):
        out[(name, "frozen")] = {"as run": R[(fam, BASE)], "no cap": R[(fam, key({"max_size": None}))],
                                 "cap 1.0": R[(fam, key({"max_size": 1.0}))], "30 bp": R[(fam, key(k30))]}
    for factor in ("mom", "mkt"):
        cost = e.Costs(kappa={factor: k30["kappa"]})
        out[(factor, "bsc")] = {"as run": crises.managed(panel, factor, "bsc"), "cap 1.5": crises.managed(panel, factor, "bsc", 1.5),
                                "cap 1.0": crises.managed(panel, factor, "bsc", 1.0), "30 bp": crises.managed(panel, factor, "bsc", costs=cost)}
    cost = e.Costs(kappa={"mom": k30["kappa"]})
    out[("mom", "frozen")] = {"as run": crises.managed(panel, "mom", "frozen"), "no cap": crises.managed(panel, "mom", "frozen", None),
                              "cap 1.0": crises.managed(panel, "mom", "frozen", 1.0), "30 bp": crises.managed(panel, "mom", "frozen", costs=cost)}
    return out


def crisis_tables(panel: pd.DataFrame, C: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three questions for every episode and every run that holds a position through its window;
    the daily paths of those windows; momentum's two crashes."""
    rows, paths = [], []
    P = {k: crises.path(*v["as run"]) for k, v in C.items()}                # once per run, not per episode
    monthly = {k: {name: runs.scored(r, per)["excess"] for name, (_, r, per) in v.items()} for k, v in C.items()}
    bench = {k: runs.scored(*v["as run"][1:])["bench_excess"] for k, v in C.items()}
    for episode, (first, last) in crises.EPISODES.items():
        s = crises.span(panel, first, last)
        for (book, version), variants in C.items():
            b, res, periods = variants["as run"]
            p = P[(book, version)]
            if not crises.covers(p, s, panel.index):
                continue
            q = crises.questions(p, s)
            rows.append({"episode": episode, "book": book, "version": version, "peak": s["peak"].date(), "trough": s["trough"].date(),
                         "market_fall": s["market_fall"], **q, "window_bench": crises.window_growth(bench[(book, version)], s["months"]),
                         **{f"window: {name}": crises.window_growth(x, s["months"]) for name, x in monthly[(book, version)].items()},
                         "window_capped_share": res.loc[res.index.isin(s["months"]), "capped"].mean()})
            w = p.loc[s["start"] - pd.DateOffset(months=2):s["end"]]          # two months more: the realized volatility's warm-up
            w = w.assign(episode=episode, book=book, version=version)
            paths.append(w.reset_index(names="date"))
    crashes = []
    for name, (lo, hi, published) in crises.MOMENTUM_CRASHES.items():
        months = pd.period_range(lo, hi, freq=MONTH)
        row = {"crash": name, "published_wml": published}
        for version in ("bsc", "frozen"):
            _, res, periods = C[("mom", version)]["as run"]
            m = runs.scored(res, periods)
            if months.isin(m.index).all():
                row |= {"bench": np.prod(1 + m.loc[months, "bench_excess"]) - 1, f"{version}": np.prod(1 + m.loc[months, "excess"]) - 1,
                        f"{version}_size_first": res.loc[months[0], "size"], f"{version}_size_mean": res.loc[months, "size"].mean()}
        crashes.append(row)
    return pd.DataFrame(rows), pd.concat(paths, ignore_index=True), pd.DataFrame(crashes)


# ---------------------------------------------------------------- 7: information

def risk_table(panel: pd.DataFrame, rec: dict, months: dict) -> pd.DataFrame:
    """Each asset's share of buy-and-hold's variance, over the recipes held in the base's scored years."""
    rows = []
    for fam, base in (("screened", "primary"), ("fixed", "fixed")):
        years = set(months[base].year)
        for method in ("inv_vol", "equal"):
            s = checks.risk_shares(panel, rec[(fam, None)], method)
            s = s[[d.year + 1 in years for d in s.index]]
            held = s.where(s != 0)
            rows += [{"family": fam, "method": method, "asset": a, "years_held": int(held[a].notna().sum()),
                      "mean_share": held[a].mean(), "max_share": held[a].max()} for a in s.columns if held[a].notna().any()]
    return pd.DataFrame(rows)


def etf_stress(panel: pd.DataFrame) -> pd.DataFrame:
    s = tracks.specs(panel)["etf"]
    rows = []
    for name, costs in (("base: SPY short fee 0.25% a year", e.COSTS["etf"]), ("stress: SPY short fee 1% a year", e.COSTS["etf_short_stress"])):
        periods = execution.holding_periods(s["book"], costs)
        res = runs.run(runs.prepare(s["book"], s["forecaster"]), 1.0, periods, min_pairs=s["min_pairs"])
        rows.append({"costs": name, **runs.metrics(res, periods)})
    return pd.DataFrame(rows)


def option_c(panel: pd.DataFrame, rec: pd.DataFrame, R: dict, months: pd.PeriodIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The primary under Entry 10's option (c) beside the pre-registered (a), and each holding year's
    scale. Information: computed after Entry 10 fixed the rule, and it changes no development verdict."""
    _, res, periods = R[("screened", BASE)]
    m = runs.scored(res, periods, months)
    scale = screened.year_scale(panel, rec, checks.METHOD)
    rows = []
    for name, x in (("(a) every month as it is (pre-registered, Entry 6)", m),
                    ("(c) each holding year scaled to an equal risk budget (Entry 10)", runs.year_weighted(m, scale))):
        test = ev.sharpe_gain_test(x)
        rows.append({"rule": name, "sharpe": ev.sharpe(x["excess"]), "bh_sharpe": ev.sharpe(x["bench_excess"]),
                     **{k: test[k] for k in ("gain", "se", "z", "p", "lower", "upper", "mde", "months", "reading")}})
    years = m.index.year
    w = runs.year_weighted(m, scale)
    share = lambda x: ((x - x.mean()) ** 2).groupby(years).sum() / ((x - x.mean()) ** 2).sum()
    held = rec.xs(checks.METHOD, level="method")["assets"]
    held.index = held.index.year + 1
    per_year = pd.DataFrame({"assets": held, "sigma_hat": 1 / scale, "scale": scale}).loc[sorted(set(years))]
    per_year["months"] = pd.Series(years).value_counts().sort_index()
    per_year["bh_realized_vol"] = m["bench_excess"].groupby(years).std() * np.sqrt(12)
    for col, name in (("bench_excess", "bh"), ("excess", "timed")):
        per_year[f"share_of_{name}_variance: (a)"], per_year[f"share_of_{name}_variance: (c)"] = share(m[col]), share(w[col])
    return pd.DataFrame(rows), per_year.rename_axis("year")


# ---------------------------------------------------------------- main

def span(m: pd.PeriodIndex) -> str:
    return f"{len(m)} months, {m[0]} to {m[-1]}" if len(m) else "no whole months"


def run_line(fam: str, spec: Spec, b: e.Book, res: pd.DataFrame, base: pd.PeriodIndex) -> str:
    """One run in the mechanics report: its periods, segments and whole months (dates and counts only)."""
    w = checks.whole_months(b, res)
    return (f"{fam:9s} {str(checks.changes(spec)):45s} periods {len(res):6d} segments {int(res['entry'].sum())} "
            f"whole months: {span(w)}; base months not scored: {len(base.difference(w))}")


def mechanics(R: dict, setup_: dict, panel: pd.DataFrame) -> None:
    """Dates and counts only: no return statistic is printed."""
    months = base_months(R)
    for base, m in months.items():
        print(f"{base}: {span(m)}")
    for (fam, spec), (b, res, periods) in sorted(R.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        print(run_line(fam, spec, b, res, months["primary" if fam == "screened" else "fixed"]))
    for episode, (first, last) in crises.EPISODES.items():
        s = crises.span(panel, first, last)
        print(f"{episode}: peak {s['peak'].date()}, trough {s['trough'].date()}, window {s['months'][0]} to {s['months'][-1]}")


def main(jobs: int = -2, only_mechanics: bool = False) -> None:
    t0 = time.time()
    panel, ohlc = load_dev(), load_etf_ohlc_dev()
    S = setup(panel, ohlc, jobs)
    print(f"books: {len(S['rec'])} recipe sets; held: {S['assets']} ({time.time() - t0:.0f}s)", flush=True)
    fill_cache(panel, S["rec"], S["assets"], jobs)
    R = all_runs(panel, S["rec"], S["assets"], jobs)
    print(f"{len(R)} runs ({time.time() - t0:.0f}s)", flush=True)
    if only_mechanics:
        mechanics(R, S, panel)
        return

    months = base_months(R)
    T, V, series = table(R, S["assets"], months)
    print(f"robustness table: {len(T)} rows ({time.time() - t0:.0f}s)", flush=True)
    P, draws = placebos(panel, R, months, jobs)
    print(f"placebo ({time.time() - t0:.0f}s)", flush=True)
    C = crisis_runs(panel, R)
    Q, paths, crashes = crisis_tables(panel, C)
    print(f"crises: {len(Q)} episode-runs ({time.time() - t0:.0f}s)", flush=True)
    risk, stress = risk_table(panel, S["rec"], months), etf_stress(panel)
    weighted, scales = option_c(panel, S["rec"][("screened", None)], R, months["primary"])

    T.to_csv(REPORTS_DIR / "phase8_robustness.csv", index=False)
    V.to_csv(REPORTS_DIR / "phase8_checks.csv", index=False)
    P.to_csv(REPORTS_DIR / "phase8_placebo.csv", index=False)
    Q.to_csv(REPORTS_DIR / "phase8_crises.csv", index=False)
    crashes.to_csv(REPORTS_DIR / "phase8_crashes.csv", index=False)
    risk.to_csv(REPORTS_DIR / "phase8_risk_shares.csv", index=False)
    stress.to_csv(REPORTS_DIR / "phase8_etf_stress.csv", index=False)
    weighted.to_csv(REPORTS_DIR / "phase8_year_weighted.csv", index=False)
    scales.to_csv(REPORTS_DIR / "phase8_year_scales.csv")
    wfb.long_table(series, ["base", "check", "variation"]).to_parquet(PROCESSED_DIR / "phase8_runs.parquet")
    paths.to_parquet(PROCESSED_DIR / "phase8_paths.parquet")
    pd.concat({f: d.set_axis(d.columns.astype(str), axis=1) for f, d in draws.items()}, names=["family"]).to_parquet(
        PROCESSED_DIR / "phase8_placebo_draws.parquet")
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=-2, help="parallel workers (joblib n_jobs; default all but one)")
    ap.add_argument("--mechanics", action="store_true", help="stop after the runs and print only dates and counts")
    args = ap.parse_args()
    main(args.jobs, args.mechanics)
