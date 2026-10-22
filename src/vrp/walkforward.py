"""
Build Track C's C2: the development walk-forward of C1
======================================================
    python -m src.vrp.walkforward      development data only; writes reports/phase10_trackc_c2_*.csv

Steps (rules in C1, PREREGISTRATION Entry 21):
  1. μ̂ from the twin: the market factor's M4 VRP and next-month excess returns, from January 1990 (first μ̂ in 1995)
  2. three runs through Phase 7's runner, each with c_t from 36 pairs and the cap of 1.5:
       c1_spy    C1 on SPY after its costs: the scored run
       c1_twin   C1 on the market factor at SPY's costs, its own M2 (beside)
       vol_spy   the volatility-only SPY rule, M2 and c_t / σ̂² (beside)
  3. c1_spy: the primary test (Sharpe gain over SPY at size 1, after costs) and Entry 6's metrics on its months
  4. what VRP adds: c1_spy against vol_spy on the same months (Ledoit-Wolf, one-sided; information)
  5. Entry 8's crisis questions on the development crises the runs cover (2000-2002, 2008-2009)

Writes phase10_trackc_c2_results.csv (one row per run), _vs_volonly.csv, _sizes.csv (the monthly μ̂, σ̂², c_t and
sizes), _crises.csv and _summary.csv.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import REPORTS_DIR
from src.data.load import load_dev
from src.engine import evaluation as ev, execution
from src.robustness import crises
from src.vrp import C1, strategy
from src.walkforward import runs

NAMES = {"c1_spy": "C1 on SPY, after costs (decides)", "c1_twin": "C1 on the twin, at SPY's costs (beside)",
         "vol_spy": "Volatility-only SPY rule, on C1's months (beside)"}


def run_all(panel: pd.DataFrame, mu: pd.Series) -> dict[str, tuple]:
    out = {}
    for k, s in strategy.specs(panel, mu).items():
        periods = execution.holding_periods(s["book"], s["costs"])
        res = runs.run(runs.prepare(s["book"], s["forecaster"]), 1.0, periods, max_size=C1["cap"], min_pairs=s["min_pairs"])
        out[k] = (s["book"], res, periods)
    return out


def results(R: dict, months: pd.PeriodIndex) -> pd.DataFrame:
    rows = []
    for k, (book, res, periods) in R.items():
        r = res[np.asarray(runs.month_of(res, periods).isin(months))]
        rows.append({"run": k, "name": NAMES[k], **runs.metrics(res, periods, months),
                     "share_size_zero": float((r["size"] == 0).mean())})
    return pd.DataFrame(rows)


def vs_volonly(R: dict, months: pd.PeriodIndex) -> pd.DataFrame:
    """C1 against the volatility-only rule, both on SPY after costs, on the same months: the Sharpe difference by
    the primary test's method (information)."""
    a = runs.scored(R["c1_spy"][1], R["c1_spy"][2], months)["excess"]
    b = runs.scored(R["vol_spy"][1], R["vol_spy"][2], months)["excess"]
    gain, se = ev.sharpe_difference(a, b)
    return pd.DataFrame([{"months": len(months), "c1_sharpe": ev.sharpe(a), "volonly_sharpe": ev.sharpe(b),
                          "gain": gain, "se": se, "p": float(norm.sf(gain / se)),
                          "size_correlation": R["c1_spy"][1]["size"].reindex(months).corr(R["vol_spy"][1]["size"].reindex(months))}])


def sizes(M: pd.DataFrame, R: dict) -> pd.DataFrame:
    c1, vol = R["c1_spy"][1], R["vol_spy"][1]
    out = pd.DataFrame({"c1_c": c1["c"], "c1_size": c1["size"], "volonly_sigma2": vol["forecast"], "volonly_c": vol["c"],
                        "volonly_size": vol["size"]})
    decided = M.copy()
    decided.index = decided.index + 1                          # μ̂ made at t's close sizes month t + 1
    return out.join(decided.add_prefix("decided_"), how="left").rename_axis("month").reset_index()


def crisis_table(panel: pd.DataFrame, R: dict) -> pd.DataFrame:
    rows = []
    paths = {k: crises.path(*v) for k, v in R.items()}
    for episode, (first, last) in crises.EPISODES.items():
        s = crises.span(panel, first, last)
        for k, p in paths.items():
            row = {"episode": episode, "run": k, "covered": crises.covers(p, s, panel.index)}
            if row["covered"]:
                row |= crises.questions(p, s)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    panel = load_dev()
    M = strategy.twin_mu(panel)
    R = run_all(panel, M["mu"])
    months = runs.scored(R["c1_spy"][1], R["c1_spy"][2]).index
    for k in ("c1_twin", "vol_spy"):
        have = runs.scored(R[k][1], R[k][2]).index
        if not months.isin(have).all():
            raise ValueError(f"{k} does not hold every month C1 on SPY holds")
    out = {"results": results(R, months), "vs_volonly": vs_volonly(R, months), "sizes": sizes(M, R),
           "crises": crisis_table(panel, R)}
    T = out["results"].set_index("run")
    out["summary"] = pd.DataFrame({"value": {
        "first_mu": str(M.index[0]), "first_sized_month": str(months[0]), "last_month": str(months[-1]),
        "months": len(months), "gain": T.loc["c1_spy", "gain"], "p": T.loc["c1_spy", "p"],
        "mde": T.loc["c1_spy", "mde"], "reading": T.loc["c1_spy", "reading"], "seconds": round(time.time() - t0)}})
    for k, df in out.items():
        df.to_csv(REPORTS_DIR / f"phase10_trackc_c2_{k}.csv", index=(k == "summary"))
    print(f"C2: {len(months)} months, {months[0]} to {months[-1]}; reading {T.loc['c1_spy', 'reading']} "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
