"""
Track B: fixes to the timing rule (Phase 10)
============================================
Four changes to the frozen rule, each aimed at a cause found in Phase 9, scored on the Sharpe ratio with the
primary test as before (Entry 3). Small gains at best are expected, so "inconclusive" on Holdout B.

    rules   the variants' sizes from the frozen engine's decisions, and the month-end signals B3 and B4 read
    build   `python -m src.fixes.build` runs the development walk-forward and reads the family of four through
            Romano-Wolf and Holm; `--mechanics` runs it end to end and prints counts only

Rules in TRACK_B, written before any Track B result on any data (PREREGISTRATION Entry 17), with the user's calls
of Entry 20: SPY is the primary book, and B1 and B2 scale by inverse volatility, as Wang and Yan do. Verified on
simulated data (test/test_fixes.py).
"""

from __future__ import annotations

from src.frozen import freeze

TRACK_B = freeze({
    "variants": {
        "B1": "fixed blend: 0.5 + 0.5 x c_t / forecast volatility (inverse volatility, its own c_t), the managed half "
              "capped at 1.5",
        "B2": "downside fixed blend: B1 on downside volatility (the forecast x the down-day share of the last 22 days' "
              "Σ r², square-rooted), its own c_t",
        "B3": "conditional targeting: size 1, unless the forecast is in the top or bottom 20% of its last 120 "
              "month-end values (itself included); then the frozen size",
        "B4": "trend gate: if the book's value at size 1 closes the month above the mean of its last 10 month-end "
              "values (that one included), the size is at least 1; below it, the frozen size",
    },
    "fixes": {"B1": "unstable mix weights", "B2": "unstable mix weights, plus return timing",
              "B3": "noise and turnover in normal months", "B4": "missed rebounds"},
    "sources": {"B1": "Wang and Yan", "B2": "Wang and Yan", "B3": "Bongaerts, Kang and van Dijk (2020)",
                "B4": "the common 10-month trend rule"},
    "grade_b": {"B1": "exploratory", "B2": "clean", "B3": "clean", "B4": "exploratory"},   # Entry 20
    "power": {"frozen": 1.0, "B1": 0.5, "B2": 0.5, "B3": 1.0, "B4": 1.0},   # size = c_t x forecast^-power. B3, B4:
                                                                            # the usual rule is the frozen one
    "cap": 1.5,
    "blend": 0.5,                        # B1, B2: the share of the account held at size 1
    "extreme": 0.2,                      # B3: the top and bottom 20% of the forecast's own recent history
    "window": 120,                       # B3: month-end forecasts; fewer at first, but at least the book's c_t minimum
    "trend_months": 10,                  # B4
    "statistic": "the primary test: the Sharpe gain over buy-and-hold, Ledoit-Wolf with Newey-West, one-sided 5% "
                 "(Entry 3); option (c) decides for a screened book (Entry 10), (a) otherwise",
    "family": ["B1", "B2", "B3", "B4"],
    "adjust": ["romano_wolf", "holm"],   # Phase 7's: stationary bootstrap, blocks 6 and 12, the larger p
    "dev_crises": ["1929-1932", "October 1987", "2000-2002", "2008-2009"],   # robustness.crises.EPISODES: B4's design
    "pass": "a variant moves to Holdout C only if its development gain survives Romano-Wolf and Holdout B does not "
            "read 'no meaningful edge'. Holdout C decides",
})

# The books Track B can run on (the plan's recommendation: SPY primary, the paper mix secondary, the twin as
# SPY's main development sample).
BOOKS = freeze({
    "spy": "SPY alone after costs: M2 from the 12th month, c_t from 36 pairs, SPY's costs (Phase 9's ETF track)",
    "twin": "the market factor at SPY's costs: M2, c_t from 120 pairs, 1946-2018 (Phase 6's SPY reference)",
    "paper": "Phase 9's primary: the real-time screened book, inverse-volatility mix, M3, c_t from 120 pairs, no costs",
})

# The user's calls before any run (PREREGISTRATION Entry 17, Section 6), made in Entry 20.
OPEN = freeze({
    "primary_book": "spy",               # development read on the twin (1946-2018); SPY's own months and the paper mix beside
    "b1_grade_b": "exploratory",         # inverse volatility was chosen after Phase 9, and it is milder than the rule that lost
})


def require_decided() -> None:
    missing = [k for k, v in OPEN.items() if v is None]
    if missing:
        raise RuntimeError(f"Track B is not frozen: decide {', '.join(missing)} (PREREGISTRATION Entry 17) before any run")
