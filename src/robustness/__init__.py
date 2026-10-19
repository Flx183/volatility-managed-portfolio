"""
Phase 8: robustness checks
==========================
Is the result sturdy or fragile? Each check changes one thing from the chosen specification and
records every outcome, good or bad. Nothing here replaces the primary test (PREREGISTRATION Entry 1,
Section 6): the checks are a map of how the result moves.

    checks     the books, every one-at-a-time variation, "time each, then combine", the placebo
    crises     the four development crises: daily paths, the three questions, momentum's crashes
    build      `python -m src.robustness.build` runs everything and writes reports/phase8_*.csv

Three bases, every check on each (Entry 8):
    primary        the pre-registered primary (Entry 6): the real-time screened book, inverse-
                   volatility mix, M3 GJR-GARCH, inverse-variance timing, monthly, cap 1.5, no costs
    primary_1997   the same runs scored on 1997-2018 only: Entry 7's cut without the market-only
                   year 1984 (information)
    fixed          Phase 6's fixed book, market + momentum + BAB from 1950, the same specification
                   (information): the book the plan's checks were written for

Every paper result is an upper bound: paper factors cannot be traded as built. Development sample only.
"""

from __future__ import annotations

from src import variance as v
from src.engine.data import DAY, MONTH, WEEK
from src.frozen import freeze

BASES = freeze({
    "primary": "the primary (screened book, Entry 6), all its scored months",
    "primary_1997": "the primary, 1997-2018 only (Entry 7's cut; information)",
    "fixed": "Phase 6's fixed book: market + momentum + BAB (information)",
})

# Fixed on 2026-10-19 before any Phase 8 result (PREREGISTRATION Entry 8).
ROBUST_RULES = freeze({
    "eras": v.SUBPERIODS,                    # Phase 2's three eras
    "min_era_months": 60,                    # an era with fewer scored months is reported, not read
    "daily_min_pairs": 2520,                 # c_t of a daily book: ten years of 252 trading days
    "windows": {"rolling 10 years": (2520, 120), "rolling 20 years": (5040, 240)},   # (GARCH days, c_t pairs)
    "caps": [1.0, 2.0, None],                # beside the base's 1.5
    "kappas": [5e-4, 10e-4, 20e-4, 30e-4],   # one-way, per unit traded, every instrument, both runs
    "borrow_spread": 0.005,                  # the all-frictions row: 30 bp and borrowed cash at rf + 0.5%
    "placebo_draws": 1000,                   # seeds 0-999, as in Phases 6 and 7
    "worst_days": 5,                         # crises: buy-and-hold's five worst days inside the span
    "lead_months": 6,                        # crises: the window opens 6 months before the peak ...
    "after_months": 12,                      # ... and closes 12 months after the trough
})

# Check -> (the base's own setting, then each variation as (label, change)). Leave-one-out is set
# per base in checks.variations, from the assets that base holds.
CHECKS = freeze({
    "subperiod": ("all months", [(era, {"era": era}) for era in v.SUBPERIODS]),
    "frequency": ("monthly", [("weekly", {"freq": WEEK}), ("daily", {"freq": DAY})]),
    "window": ("expanding", [(k, {"window": k}) for k in ROBUST_RULES["windows"]]),
    "cap": ("1.5", [(f"{c:.1f}" if c else "none", {"max_size": c}) for c in ROBUST_RULES["caps"]]),
    "rule": ("inverse variance", [("inverse volatility", {"rule": "inv_vol"}), ("downside volatility", {"rule": "down_vol"})]),
    "order": ("combine, then time", [("time each, then combine", {"order": "each"})]),
    "leave_one_out": ("none dropped", []),
    "costs": ("0 bp", [(f"{k * 1e4:.0f} bp", {"kappa": k}) for k in ROBUST_RULES["kappas"]]
              + [("30 bp + borrowing at rf + 0.5%", {"kappa": 30e-4, "borrow": ROBUST_RULES["borrow_spread"]})]),
})
FREQS = {MONTH: "monthly", WEEK: "weekly", DAY: "daily"}

from src.robustness import checks, crises     # noqa: E402  (after the rules they read)
