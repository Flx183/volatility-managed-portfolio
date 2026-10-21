"""
Phase 9: the final holdout test
===============================
One honest, untouched test. It runs once, on the sealed holdout (2019-01-01 onward), with the code,
models, parameters and cutoffs frozen at the commit tagged v1-frozen (PREREGISTRATION Entry 13).

    setup    the development panel joined to the holdout; where each book's data ends; the books and
             their frozen specifications; the crisis spans
    build    `python -m src.holdout.build` opens the holdout once, runs everything and writes
             reports/phase9_*.csv. `--rehearse` runs the same code on development data only, with a
             pseudo-holdout from 2011, and never opens the holdout

What runs (Entry 13):
    primary    the pre-registered primary (Entry 6): the real-time screened book, inverse-volatility mix,
               M3 GJR-GARCH, inverse-variance timing, monthly, cap 1.5. Option (c) decides its reading,
               the unscaled (a) is reported beside it (Entry 10)
    best       Phase 7's best variant, the family's largest development gain: M0, downside-volatility
               timing, shrunk mean-variance mix, weekly
    spy, etf   the ETF track, after costs: SPY alone (M2) and SPY + MTUM's tilt (M3) (Entry 6)
    fixed      Phase 6's fixed book, market + momentum + BAB (information)

Every run is the development walk-forward continued: the same code on the joined panel, so each decision
reads only data up to its own close, and the account carries its December 2018 position into January 2019.
"""

from __future__ import annotations

from src.frozen import freeze
from src.walkforward import PRIMARY

# Fixed on 2026-10-20, before the holdout was opened (PREREGISTRATION Entry 13).
HOLDOUT_RULES = freeze({
    "start": "2019-01-01",                       # config.HOLDOUT_START: the first holdout day
    "data": "the sealed snapshot (data/holdout_lock.json), last day 2026-08-31; not refreshed",
    "paper_end": "the last month in which every factor has a return on every trading day",
    "etf_end": "the last month in which SPY and MTUM have a return on every trading day",
    "year_end": "the last trading day of December; the data's last day is not a year-end",
    "primary": PRIMARY,                          # (model, rule, mix, frequency)
    "best": ("M0", "down_vol", "shrunk_mv", "weekly"),   # phase7_summary.csv: best_variant
    "decides": {"primary": "(c)", "best": "(c)", "spy": "(a)", "etf": "(a)", "fixed": "(a)"},
    "scoring": "calendar months from the daily ledger (src/robustness/ledger.py), at every frequency",
    "fixed_from": "the recipe set at the last year-end before the holdout",   # information: held from the start
    "placebo_draws": 1000,                       # the primary only, read on the holdout months (information)
    "combination_train": "every scored month before the month (development included)",
    "crisis_runs": ["primary", "best", "fixed", "spy", "etf", "mom frozen", "mom bsc", "mkt bsc"],
})

# Episode -> (the first day the market's peak is searched from, the year the trough is searched in).
# Trough: the market's lowest close (total return) in that year. Peak: its highest close from the first
# day up to the trough. The window runs from 6 months before the peak to 12 months after the trough
# (Entry 8's lead and after months).
EPISODES = freeze({"2020": ("2020-01-01", "2020"), "2022": ("2021-01-01", "2022")})

# The plan's table for reading the outcome. A holdout reading (Entry 3) maps to the plan's three columns;
# development is "wins" only if its own registered test read "edge".
HOLDOUT_OUTCOME = freeze({"edge": "wins", "inconclusive": "flat, slightly negative or too short to tell",
                          "no meaningful edge": "clearly negative"})
VERDICT = freeze({
    ("wins", "wins"): "evidence of a real edge; still modest, given the short holdout",
    ("wins", "flat, slightly negative or too short to tell"): "inconclusive: too short to tell; report both",
    ("wins", "clearly negative"): "likely overfit, or the edge faded; report it",
    ("fails", "wins"): "consistent with Cederburg et al.; report it",
    ("fails", "flat, slightly negative or too short to tell"): "consistent with Cederburg et al.; report it",
    ("fails", "clearly negative"): "consistent with Cederburg et al.; report it",
})
# Each book's development result under its own registered rule (Entries 5 and 7), fixed before the holdout.
DEVELOPMENT = freeze({
    "primary": ("fails", "Phase 7's primary test: +0.013, p 0.471, inconclusive (option (c), information: +0.214, p 0.059)"),
    "best": ("fails", "Phase 7's family: +0.234, raw p 0.089, inconclusive; Romano-Wolf p 0.43; deflated Sharpe 0.67"),
    "spy": ("fails", "Phase 7's ETF track: +0.165, p 0.13, inconclusive"),
    "etf": ("fails", "Phase 7's ETF track: +0.088, p 0.30, inconclusive (20 months)"),
    "fixed": ("wins", "Phases 6-7: +0.355, p 0.0002, edge, on an asset list chosen with the whole development sample"),
})
NAMES = freeze({"primary": "Primary (screened book, M3)", "best": "Best variant (M0, downside vol, shrunk MV, weekly)",
                "spy": "ETF track: SPY alone (M2)", "etf": "ETF track: SPY + MTUM tilt (M3)",
                "fixed": "Phase 6's fixed book (M3, information)"})

# The rehearsal: the same build on development data only. Its pseudo-holdout starts in 2011, BAB loses its
# last month (as AQR's file does in the real holdout), and two development drawdowns stand in for the crises.
REHEARSAL = freeze({"start": "2011-01-01", "drop_last_month": "bab",
                    "episodes": {"2011": ("2011-01-01", "2011"), "2015-2016": ("2015-01-01", "2016")}})
