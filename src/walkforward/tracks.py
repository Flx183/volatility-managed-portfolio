"""
Phase 7: the other books
========================
None of these is the primary test, and nothing in them changes a choice (PREREGISTRATION Entry 6).

The ETF track. No choice is made on ETF data: the frozen paper-factor choices are re-run on what can be
traded, at the ETF costs (Entry 2), with inverse-variance timing, the 1.5 cap and monthly rebalancing.
    spy      SPY timed alone with M2 GARCH(1,1), the SPY reference's frozen model
    etf      SPY plus MTUM's market-hedged tilt (Phase 4's ETF recipe, inverse-volatility mix) with M3
             GJR-GARCH, the combined portfolio's frozen model
Both make their first forecast at the end of the mix's 12th month (the ETF minimum of Phases 3 and 4)
and need 36 pairs for c_t (Entry 3), so SPY is timed from 1997 and the ETF book from May 2017. Their
development window is short; the holdout is their real test.

Two long-history references, both already scored in Phase 6:
    market   the market factor at SPY's costs with M2 (Phase 6's SPY reference), 1946-2018
    fixed    Phase 6's fixed book (market + momentum + BAB, the full-sample asset list) with M3: the
             distance between it and the primary is what selecting assets in real time costs
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import engine as e
from src import models as md
from src import portfolio as pf
from src.models import filters, inputs
from src.models import scorecard as sc

ETF_FIRST_MONTH = 12        # the ETF track's first forecast: the end of the mix's 12th month
ETF_MIN_PAIRS = e.ENGINE_RULES["etf_min_months"]      # 36


def etf_garch(gjr: bool, months: int = ETF_FIRST_MONTH):
    """M2 (gjr=False) or M3 with the first forecast at the end of the mix's `months`-th month: Phase 6's
    walk-forward (filters.garch_fits: refit at every month-end from the last estimate, never ending at a
    worse likelihood) with a shorter start."""
    def forecaster(p: pd.Series) -> pd.Series:
        p = p.dropna()
        r = p.to_numpy() * filters.SCALE
        r2 = r ** 2
        neg = np.where(r < 0, r2, 0.0)
        o = inputs.origins(p).iloc[months - 1:]
        out, theta = [], None
        for pos in o.to_numpy():
            n = pos + 1
            init = r2[:n].mean()
            theta, _ = filters.garch_fit(r2[:n], neg[:n], init, gjr, theta)
            s2 = filters.garch_path(theta, r2[:n], neg[:n], init)[-1]
            w, a, g, b = theta
            out.append(filters.garch_sum(theta, w + a * r2[n - 1] + g * neg[n - 1] + b * s2) / filters.SCALE ** 2)
        return pd.Series(out, index=o.index, name="forecast")
    forecaster.__name__ = f"{'M3' if gjr else 'M2'}_from_month_{months}"
    return forecaster


def specs(panel: pd.DataFrame) -> dict[str, dict]:
    """Each book with its forecaster, costs and c_t minimum."""
    fs = md.forecasters(panel, down=False)                 # M2 and M3 do not depend on `down`
    phase6 = sc.books(panel)
    return {
        "spy": {"label": "ETF track: SPY alone, M2", "book": e.Book.single(panel["spy"], panel["rf"], "spy"),
                "forecaster": etf_garch(False), "costs": e.COSTS["etf"], "min_pairs": ETF_MIN_PAIRS},
        "etf": {"label": "ETF track: SPY + MTUM tilt, M3",
                "book": e.data.from_phase4(panel, pf.recipes(panel, "etf"), pf.PRIMARY[1]),
                "forecaster": etf_garch(True), "costs": e.COSTS["etf"], "min_pairs": ETF_MIN_PAIRS},
        "market": {"label": "Market factor at SPY's costs, M2 (Phase 6's SPY reference)", "book": phase6["spy"]["book"],
                   "forecaster": fs["M2"], "costs": sc.SPY_COSTS, "min_pairs": e.ENGINE_RULES["min_months"]},
        "fixed": {"label": "Phase 6's fixed book (market + momentum + BAB), M3", "book": phase6["paper"]["book"],
                  "forecaster": fs["M3"], "costs": e.COSTS["paper"], "min_pairs": e.ENGINE_RULES["min_months"]},
    }
