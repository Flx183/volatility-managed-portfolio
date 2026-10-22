"""
Build Track C's Phase C0: the EDA and its gate
==============================================
    python -m src.vrp.build      development data only; writes reports/phase10_trackc_*.csv

Steps:
  1. the market factor's M0, M4 and M6 forecasts (Phase 6's walk-forward, cached), and the PutWrite index's
     development rows (src/vrp/putwrite.py; sealed on first use)
  2. VRP from each model, January 1990 to December 2018
  3. question 1: its level, sign, persistence and its months around 1998, 2001 and 2008
  4. questions 2 and 3, and the gate: the market's excess return 1, 3 and 6 months ahead on each VRP (9 tests,
     Holm), the slope in three equal eras, the out-of-sample R²; SPY's own months beside (information)
  5. question 4: each model's accuracy against its VRP's strength
  6. question 5: the PutWrite index's excess returns on each VRP, January 2007 to December 2018 (information)

Writes phase10_trackc_vrp.csv (the monthly series), _describe.csv, _episodes.csv, _tests.csv, _accuracy.csv,
_gate.csv and _summary.csv.
"""

from __future__ import annotations

import time

import pandas as pd

from src import models as md
from src.config import REPORTS_DIR
from src.data.load import load_dev
from src.engine import data
from src.vrp import TRACK_C, eda, putwrite


def main() -> None:
    t0 = time.time()
    panel = load_dev()
    fs = md.forecasters(panel, False)                       # M6 on M4's inputs: the market fails Test 3 (Entry 5)
    F = {m: fs[m](panel[TRACK_C["returns"]]) for m in TRACK_C["models"]}
    V = eda.vrp(panel, F)
    print(f"VRP: {len(V)} months, {V.index[0]} to {V.index[-1]} ({time.time() - t0:.0f}s)", flush=True)

    putwrite.fetch()
    print(putwrite.prepare(), flush=True)

    out = {"vrp": V.rename_axis("month").reset_index(), "describe": eda.describe(V), "episodes": eda.episodes(V)}

    H = TRACK_C["horizons"]
    series = {"mkt": data.monthly(panel["mkt"], panel["rf"]), "spy": data.monthly(panel["spy"], panel["rf"]),
              "put": eda.monthly_put(putwrite.load_dev(), panel)}
    T = []
    for name, mon in series.items():
        t = eda.tests(V, {h: eda.ahead(mon, h) for h in H})
        T.append(t.assign(returns=name, decides=name == TRACK_C["returns"]))
    T = pd.concat(T, ignore_index=True)
    T.loc[~T["decides"], "p_holm"] = float("nan")            # Holm is the gate's, across the market's 9 tests only
    out["tests"] = T

    G, passed = eda.gate(T[T["decides"]])
    out["gate"] = G
    A = eda.accuracy(F, panel["mkt"], V.index)
    strength = T[T["decides"]].groupby("model")[["t", "r2_oos"]].mean().add_prefix("mean_")
    out["accuracy"] = A.set_index("model").join(strength).reset_index()
    out["summary"] = pd.DataFrame({"value": {"months": len(V), "first": str(V.index[0]), "last": str(V.index[-1]),
                                             "gate_passes": passed, "cells_passing": int(G["passes"].sum()),
                                             "seconds": round(time.time() - t0)}})
    for k, df in out.items():
        df.to_csv(REPORTS_DIR / f"phase10_trackc_{k}.csv", index=(k == "summary"))
    print(f"gate {'PASSES' if passed else 'fails'}: {int(G['passes'].sum())} of {len(G)} cells "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
