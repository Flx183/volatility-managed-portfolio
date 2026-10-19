"""
Phase 7: the walk-forward evaluation
====================================
At every rebalance, from data up to that close only: re-screen (at each year-end), refit, forecast,
set c_t and the size, trade at that close, and record what the next period earns. Then the primary
test on its own, and every variant with honest control of false positives.

    screened   the real-time screen at every year-end, and the recipes and books it gives
    runs       a variant through the engine, segment by segment; its monthly returns and metrics
    stats      the block bootstrap, the CER and combination tests, Romano-Wolf, Holm, the deflated
               Sharpe ratio
    tracks     the ETF track and two long-history references
    build      `python -m src.walkforward.build` runs everything and writes reports/phase7_*.csv

The family: 8 forecasts x 3 timing rules x 4 mixes x 2 rebalance frequencies = 192 variants. The
primary is one of them, fixed before the data: M3 GJR-GARCH, inverse-variance timing,
inverse-volatility mix, monthly. Its p-value is read as is; the family's are adjusted.

Rules in WF_RULES and the modules' constants (PREREGISTRATION Entry 6). Development sample only.
"""

from __future__ import annotations

from src import portfolio as pf
from src.engine.data import MONTH, WEEK
from src.frozen import freeze
from src.models import ORDER
from src.walkforward import stats, screened, runs, tracks     # stats first: runs imports it

FREQS = freeze({"monthly": MONTH, "weekly": WEEK})
FAMILY = freeze({"model": ORDER, "rule": ["inv_var", "inv_vol", "down_vol"], "mix": pf.METHODS, "freq": list(FREQS)})
PRIMARY = ("M3", "inv_var", "inv_vol", "monthly")
# Timing rule -> (power, downside): size = c_t x forecast^(-power); downside rules forecast σ̂⁻² (runs.downside).
RULES = freeze({"inv_var": (1.0, False), "inv_vol": (0.5, False), "down_vol": (0.5, True)})

# Fixed on 2026-10-19 before any walk-forward result (PREREGISTRATION Entry 6).
WF_RULES = freeze({
    "screen": "all seven factors at every year-end; filters 1, 3, 6 (level φ), 120 months",
    "empty": "no survivors, or no c_t yet: cash, and the period is not scored",
    "flip_flops": "no rule: hold what passes at each year-end",
    "m6_base": "M4 (Phase 6's Test 3 on the combined portfolio)",
    "max_size": 1.5,
    "min_pairs": runs.MIN_PAIRS,                     # c_t: 120 monthly or 520 weekly pairs
    "delay": 0,                                       # trade at the decision close; the position earns from the next day
    "gamma": runs.GAMMA,
    "level": 0.05,                                    # every test one-sided at 5%; the family's error rate is 5%
    "bootstrap": stats.BOOT,                          # stationary, mean blocks 6 and 12: the larger p is reported
    "combination": stats.COMBINATION,
    "down_days": runs.DOWN_DAYS,
})


def variant_name(v: tuple) -> str:
    model, rule, mix, freq = v
    return f"{model} | {rule} timing | {mix} mix | {freq}"
