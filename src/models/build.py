"""
Build the Phase 6 results
=========================
    python -m src.models.build            fill the forecast cache, run the placebos, write the tables
    python -m src.models.build --fresh    delete every cached forecast first (a change to the model
                                          code starts a new cache by itself)

Every forecast is a walk-forward per recipe (78 recipes for the primary book), so the slow models
(GARCH refit monthly, boosting tuned yearly) run in parallel, one task per recipe. The engine then
reads them from the cache. Development sample only.

Writes reports/phase6_scorecard_a.csv, phase6_scorecard_b.csv, phase6_selection.csv and
data/processed/phase6_placebo.parquet (the 1,000 shuffled alphas of each model).
"""

from __future__ import annotations

import argparse
import shutil
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import engine as e
from src import variance as v
from src.config import PROCESSED_DIR, REPORTS_DIR
from src.data.load import load_dev
from src.models import ORDER, forecasters, inputs
from src.models import scorecard as sc

PLACEBO = PROCESSED_DIR / "phase6_placebo.parquet"
SLOW = ["M2", "M3", "M4", "M5", "M6", "M7"]        # walk-forwards worth caching


def setup(panel: pd.DataFrame) -> dict[str, dict]:
    """Each book with its forecasters. M6 and M7 build on M5 if Phase 2's Test 3 passes on the
    series being forecast, else on M4."""
    out = sc.books(panel)
    for spec in out.values():
        spec["test3"] = v.t3_asymmetry(spec["series"])
        spec["fs"] = forecasters(panel, down=spec["test3"]["pass"])
    return out


def fill_cache(spec: dict, jobs: int) -> None:
    book = spec["book"]
    ids = e.data.recipe_ids(book.positions)
    mixes = [e.data.mix(book, book.positions.loc[ids == k].iloc[0]) for k in ids.unique()]
    tasks = [(m, p) for m in reversed(SLOW) for p in reversed(mixes)]     # longest first
    Parallel(n_jobs=jobs, verbose=5)(delayed(spec["fs"][m])(p) for m, p in tasks)


def runs(spec: dict) -> dict[str, pd.DataFrame]:
    return {m: sc.run(spec["book"], spec["fs"][m], spec["costs"]) for m in ORDER}


def sanity(rs: dict[str, pd.DataFrame], limit: float = 1e3) -> None:
    """Stop if any forecast is more than `limit` times (or less than 1/limit of) M0's for the same
    month. Real forecasts stay within about 10x of each other; the first run's broken GARCH fits
    (PREREGISTRATION Entry 5) reached 5 x 10¹⁰."""
    base = rs["M0"]["forecast"]
    for m, r in rs.items():
        ratio = r["forecast"] / base
        if not (np.isfinite(ratio).all() and ratio.between(1 / limit, limit).all()):
            bad = ratio[~ratio.between(1 / limit, limit)]
            raise RuntimeError(f"{m}: forecast {bad.iloc[0]:.3g} x M0's in {bad.index[0]}; a fit has broken down")


def placebos(spec: dict, jobs: int) -> pd.DataFrame:
    """The engine's shuffled-forecast placebo (1,000 draws) for every model, at the primary specification."""
    def one(m):
        r = e.placebo(spec["book"], sc.SCORE_RULES["rule"], forecaster=spec["fs"][m], costs=spec["costs"],
                      max_size=sc.SCORE_RULES["max_size"])
        return m, r
    out = Parallel(n_jobs=jobs)(delayed(one)(m) for m in ORDER)
    return pd.DataFrame({m: {"alpha": r["alpha"], "share_beaten": r["share_beaten"], "pass": r["pass"],
                             "shuffled": r["shuffled"]} for m, r in out}).T


def main(fresh: bool = False, jobs: int = -2) -> None:
    if fresh and inputs.CACHE_ROOT.exists():
        shutil.rmtree(inputs.CACHE_ROOT)
    panel = load_dev()
    specs = setup(panel)
    tables = {"a": [], "b": [], "selection": []}
    placebo_rows = []
    for name, spec in specs.items():
        t0 = time.time()
        fill_cache(spec, jobs)
        print(f"{name}: forecasts cached in {time.time() - t0:.0f}s")
        rs = runs(spec)
        sanity(rs)
        a = sc.scorecard_a(rs, sc.targets(spec["book"]))
        b = sc.scorecard_b(rs)
        sel = sc.select(a, rs)
        pl = placebos(spec, jobs)
        b = b.join(pl[["share_beaten", "pass"]].rename(columns={"share_beaten": "placebo_share", "pass": "placebo_pass"}))
        for key, t in (("a", a), ("b", b), ("selection", sel)):
            tables[key].append(t.assign(book=name).set_index("book", append=True).swaplevel())
        placebo_rows.append(pd.DataFrame({"book": name, "model": np.repeat(pl.index, [len(s) for s in pl["shuffled"]]),
                                          "alpha_shuffled": np.concatenate(pl["shuffled"].to_numpy())}))
        print(f"{name}: chosen {sel.index[sel['chosen']][0]} ({time.time() - t0:.0f}s)")
    for key, parts in tables.items():
        pd.concat(parts).to_csv(REPORTS_DIR / f"phase6_scorecard_{key}.csv" if key != "selection"
                                else REPORTS_DIR / "phase6_selection.csv")
    pd.concat(placebo_rows).to_parquet(PLACEBO)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="empty the forecast cache first")
    ap.add_argument("--jobs", type=int, default=-2, help="parallel workers (joblib n_jobs; default all but one)")
    args = ap.parse_args()
    main(args.fresh, args.jobs)
