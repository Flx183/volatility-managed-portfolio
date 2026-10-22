"""
Track A: vol timing as a risk overlay (Phase 10)
================================================
The claim the development data point to: timing improves downside-risk-adjusted returns even when the
Sharpe ratio does not move. This is also how the industry uses it.

    H1   the timed book's primary risk metric beats buy-and-hold's
    H0   it does not

    sizing    the variants' sizes from the frozen engine's decisions, and a book run under each
    metrics   the Sortino gain test (the secondary), and the risk numbers reported beside it

Rules in TRACK_A, written before any Track A result on any data (PREREGISTRATION Entry 16), with the user's calls of
Entry 20: SPY only; A0 and A1 (A2 dropped: the blend lives on as Track B's B1); the primary metric is volatility
stability. A pass means the overlay keeps risk steadier out of sample than buy-and-hold, not that it earns more.
Verified on simulated data (test/test_overlay.py).
"""

from __future__ import annotations

from src.frozen import freeze

TRACK_A = freeze({
    "variants": {
        "A0": "the frozen SPY book unchanged: M2, size c_t / forecast variance, cap 1.5 (Phase 9's ETF track)",
        "A1": "rolling volatility target: σ̂_y / forecast volatility, cap 1.5; σ̂_y the book's buy-and-hold "
              "volatility over the 10-year window set at the last year-end",
    },
    "grade_b": {"A0": "not testable: Phase 9 printed its risk numbers; Holdout C only", "A1": "exploratory"},
    "cap": 1.5,
    "window_years": 10,                  # σ̂_y: the recipe's window (Entry 10) or, for one asset, the last 10 years
    "horizon": 22,                       # trading days a forecast covers: σ̂_y is read over the same 22 days
    "metric_options": ["sortino", "es_matched", "max_drawdown", "vol_stability"],
    # Entry 20: the primary metric. σ_m is the realized volatility of month m's daily returns, timed and buy-and-hold.
    "vol_stability": "R = SD(ln σ_m timed) / SD(ln σ_m buy-and-hold) over the scored months: scale-free. Edge if R < 1, "
                     "one-sided at 5%, by Entry 6's paired stationary bootstrap (blocks 6 and 12, the larger p). No "
                     "meaningful edge if the one-sided 95% upper bound on the reduction 1 - R is below 0.10",
    "secondary": "the Sortino gain (metrics.sortino_gain_test), no meaningful edge below 0.16",
    "fallback": "the first development step is a power check at 36 months on A1: if volatility stability's power is "
                "below 80%, the Sortino gain becomes the primary, with the 0.16 bound",
    "development": "the market factor at SPY's costs (M2, 1946-2018), as in Track B; SPY's own months beside",
    "matched": "a risk number that is not scale-free compares the timed run with buy-and-hold scaled to the "
               "timed run's realized volatility, so a smaller size alone wins nothing",
    "es_level": 0.05,
    "gamma": 5,
    "family": ["A0", "A1"],
    "adjust": "holm",
    "pass": "the primary metric reads edge on Holdout C at the formal date. Holdout B can only drop a variant (a large "
            "loss): A0 is not testable on it and A1 is exploratory, so it gives no clean evidence",
})

# The user's calls before any run (PREREGISTRATION Entry 16, Section 6), made in Entry 20.
OPEN = freeze({
    "primary_metric": "vol_stability",   # Sortino the secondary, and the fallback (TRACK_A["fallback"])
    "books": "spy",                      # SPY with M2 only: tradeable, logged before each month
    "meaningful_gain": 0.10,             # on the reduction 1 - R; 0.16 on the Sortino gain if it becomes primary
    "formal_date": "36 months after the freeze",
    "a2_grade_b": "dropped",             # A2 left the family; it would have been exploratory
})


def require_decided() -> None:
    missing = [k for k, v in OPEN.items() if v is None]
    if missing:
        raise RuntimeError(f"Track A is not frozen: decide {', '.join(missing)} (PREREGISTRATION Entry 16) before any run")
