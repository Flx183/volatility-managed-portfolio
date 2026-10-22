"""
Phase 10 on: testing after the holdout was seen
===============================================
Phase 9 opened the holdout once. From Phase 10 it is Holdout B: reused as test data only, one run per frozen
strategy, never for fitting, tuning, screening or choosing. Because it has been seen, every result on it
carries a grade, and a new sealed forward sample, Holdout C, does the final confirming (PREREGISTRATION
Entry 15).

    harness   the runner every Phase 10+ strategy goes through on Holdout B: the run-once and frozen-code
              checks, the joined panel (Phase 9's, through the logged loader), the report files, the grades

The machinery is Phase 9's, reused as is: the primary test (Sharpe gain, Ledoit-Wolf with Newey-West,
one-sided 5%), its three readings, the minimum detectable effect, Romano-Wolf and Holm, the deflated Sharpe
ratio, the placebo and the crisis questions.
"""

from __future__ import annotations

from src.frozen import freeze

# Fixed on 2026-10-21, before any Phase 10 result (PREREGISTRATION Entry 15).
RETEST_RULES = freeze({
    "development": ("1926-07-01", "2018-12-31"),          # EDA, fitting, tuning, choosing (VIX from 1990)
    "holdout_b": ("2019-01-01", "2026-08-31"),            # config.HOLDOUT_START .. HOLDOUT_END: test only, seen in Phase 9
    "holdout_c": "2026-09-01 onward",                     # config.FORWARD_START: sealed; monthly paper trading once frozen
    "b_use": "test only: never fit, tune, screen or choose on it",
    "b_runs": "one per frozen strategy: the harness refuses a second once its results file exists",
    "freeze": "code, parameters and a dated PREREGISTRATION entry, pinned to a commit: src/ unchanged since it",
    "b_mde": 0.45,                                        # Sharpe a year: about the smallest gain Holdout B can detect
    "c_log": "paper traded monthly from the freeze date, each month's position logged before the month starts",
    "c_formal_test": "one date per strategy, fixed in its PREREGISTRATION entry",
    "c_interim": "every 12 months, and only for a large failure (the no-meaningful-edge reading)",
})

# Every Holdout B result carries one of these (PREREGISTRATION Entry 15).
GRADES = freeze({
    "clean": "the idea came from development data or the papers",
    "exploratory": "the idea came from Phase 9's results: not evidence; only Holdout C can confirm it",
})

# What Phase 9 showed, and the ideas it could bias. An idea in the second column is graded exploratory.
CONTAMINATION = freeze([
    {"seen": "2020's rebound missed: SPY timed +19% against +77% in the year after the trough",
     "could_bias": "trend gates, re-entry rules"},
    {"seen": "mean size 0.47; the holdout's volatility above 1997-2018's",
     "could_bias": "explicit volatility targets, a rolling c_t"},
    {"seen": "the downside-volatility variant fell from the best on development to -0.114",
     "could_bias": "dropping or keeping downside volatility"},
    {"seen": "the screen held market, momentum and BAB at every year-end (profitability for 2020 and 2022)",
     "could_bias": "the asset list"},
    # Added with Entry 15 (Claude): what the Phase 9 files hold beyond the plan's four rows.
    {"seen": "every Phase 9 book's Sortino ratio, maximum drawdown, CER and volatility (reports/phase9_results.csv)",
     "could_bias": "the choice of a risk metric"},
    {"seen": "every Phase 9 book's monthly returns, timed and buy-and-hold (data/processed/phase9_runs.parquet)",
     "could_bias": "any variant whose returns are a fixed function of a Phase 9 run, such as a blend of the timed "
                   "run and buy-and-hold: its Holdout B result can be computed without a new run"},
])
