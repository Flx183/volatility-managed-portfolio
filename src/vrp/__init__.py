"""
Track C: the variance risk premium (Phase 10)
=============================================
Our forecasts move to a job where accuracy is the payoff: the gap between the variance options price in (VIX)
and the variance we forecast,

    VRP_t = (VIX_t / 100)² / 12 - σ̂²_{t+1}

VIX covers 30 calendar days (about 21 trading days), the forecast the next 22: a small mismatch, noted. M6 already
reads VIX, so its VRP partly nets VIX against itself; M4 is the cleaner comparison.

Phase C0 is EDA on development data only, January 1990 to December 2018. It becomes a strategy only if the gate
below, fixed before the EDA ran, passes (PREREGISTRATION Entry 18).

    putwrite   the CBOE PutWrite index: the raw file, its development rows, its Holdout B rows sealed by hash
    eda        VRP from each model, the predictive regressions, the eras, the out-of-sample R², the gate
    build      `python -m src.vrp.build` runs C0 once and writes reports/phase10_trackc_*.csv

The gate passed (Entry 19). C1, the strategy, is equity timing on SPY (Entry 20; design in Entry 21):

    strategy     μ̂ from the twin's VRP, the size c_t x μ̂ / σ̂², and the books it runs on
    walkforward  `python -m src.vrp.walkforward` runs C2, the development walk-forward, and writes
                 reports/phase10_trackc_c2_*.csv

Track C stopped after C2 (Entry 23): VRP added nothing over volatility timing on development (-0.015, p 0.56), and
Holdout B, never opened for Track C, stays clean for a later VRP idea.
"""

from __future__ import annotations

from src.frozen import freeze

# Fixed on 2026-10-22, before the EDA ran (PREREGISTRATION Entry 18). Marked (my choice) there where the plan
# left a detail open.
TRACK_C = freeze({
    "first_month": "1990-01",            # VIX's first month-end; the last is December 2018 (development)
    "models": ["M0", "M4", "M6"],        # Phase 6's: rolling 22-day RV, HAR, HAR-X (VIX and events); M6 on M4's inputs
    "horizons": [1, 3, 6],               # months ahead
    "returns": "mkt",                    # the market factor, SPY's long-history twin (SPY starts in February 1993)
    "beside": "spy",                     # SPY's own months, reported beside (information, not in the gate)
    "sign": 1,                           # the expected sign of the slope (Bollerslev, Tauchen and Zhou 2009)
    "nw_lags": "6 + h - 1",              # Newey-West: the monthly 6, plus h - 1 for the overlap of h-month returns
    "level": 0.05,                       # one-sided, Holm-adjusted across the 9 tests (3 models x 3 horizons)
    "eras": 3,                           # equal thirds of each regression's months; the slope's sign in each
    "oos_min_pairs": 60,                 # expanding window: the first prediction once 60 complete pairs exist
    "units": 1e4,                        # VRP reported in %² a month: VIX 20 alone is 33.3
    "episodes": {"1998": ("1998-06", "1998-12"), "2001": ("2001-07", "2001-12"), "2008": ("2008-07", "2009-03")},
    "gate": "for at least one model and horizon, all three: the slope has the expected sign with a Holm-adjusted "
            "p below 5% across the 9 tests; the sign holds in all three eras; the expanding-window out-of-sample "
            "R² on development is positive",
})

# The CBOE PutWrite index (question 5, information): not in the gate.
PUTWRITE = freeze({
    "url": "https://cdn.cboe.com/api/global/us_indices/daily_prices/PUT_History.csv",
    "raw": "cboe_put/PUT_History.csv",   # under data/raw/
    "lock": "putwrite_lock.json",        # under data/: the raw file's hash and the sealed Holdout B rows' hash
    "first": "the first month whose every trading day of the panel has a value (January 2007: the file's daily "
             "history starts there; the 7 rows before are stray)",
})

# C1: equity timing on SPY. The user's calls of Entry 20 and of 2026-10-22 (Entry 21), fixed before C2 ran.
C1 = freeze({
    "rule": "size = c_t x μ̂_t / σ̂²_t, capped at 1.5: Moreira and Muir's w = μ / σ², with VRP supplying μ",
    "vrp": "the twin's (the market factor's) M4 VRP, as in the EDA, in every run",
    "mu_returns": "mkt",                 # μ̂ is fitted on the twin's monthly excess returns (it warms up from 1990)
    "mu_model": "M4",
    "mu_horizon": 1,                     # next month's excess return: monthly sizing
    "mu_min_pairs": 60,                  # the gate's own out-of-sample start (TRACK_C["oos_min_pairs"]): first μ̂ January 1995
    "shrink": 0.5,                       # toward the mean of the same training window, at a fixed weight
    "floor": 0.0,                        # applied after shrinking (Campbell and Thompson 2008)
    "variance": "SPY's frozen M2 (tracks.etf_garch, from SPY's 12th month); the twin's own M2 when the twin is scored",
    "min_pairs": 36,                     # c_t, on the scored instrument's returns (the ETF track's rule)
    "cap": 1.5,
    "scored": "SPY after its costs (5 bp a trade, borrowing at rf + 0.5%), from its first sized month to December 2018",
    "statistic": "(a): the primary test, the Sharpe gain over SPY held at size 1 after the same costs (Entry 3)",
    "beside": ["the volatility-only SPY rule (M2, c_t / σ̂², 36 pairs) on the same months: what VRP adds",
               "the same rule scored on the twin (the market factor at SPY's costs, its own M2)",
               "Entry 8's crisis questions, 2000-2002 and 2008-2009"],
})
