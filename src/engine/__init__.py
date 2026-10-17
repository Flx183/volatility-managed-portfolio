"""
Phase 5: the backtest engine
============================
Five stages, each in its own module and testable alone:

    data         what is traded: daily excess returns, risk-free rate, positions per unit of size
    forecast     next month's variance, forecast at each month-end from data up to it
    weights      size = c_t x signal (1/σ̂² or 1/σ̂), c_t matched in real time, capped
    execution    trade at the month-end close, drift within the month, trading cost, financing
    evaluation   Sharpe ratio, spanning alpha, break-even cost, the shuffled-forecast placebo

    run(book, rule="inv_var")   chains them: one row per holding month
    placebo(book, rule)         real alpha against 1,000 runs with the forecast shuffled in time

A new forecast model is a function (daily returns -> monthly Series), passed as `forecaster`;
nothing else changes.

The benchmark is the same book at a constant size of 1 (the buy-and-hold mix), traded on the same
dates and paying the same costs, so the timed and untimed runs differ only in the size.

Phase 5 verifies the engine on simulated data only (src/engine/synthetic.py). Real-data strategy
results are first read in Phase 6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine import data, evaluation, execution, forecast, weights
from src.engine.data import Book
from src.engine.execution import COSTS, NO_COSTS, Costs

# ---------------------------------------------------------------- rules
# Fixed on 2026-10-18 from the research plan, before any real-data result under them (PREREGISTRATION
# Entry 2; the ETF track's 36 months, Entry 3). Phase 4's 1.5 cap carries over; Phase 4's c (a 10-year
# window, set once a year) is replaced by the real-time c_t below.
ENGINE_RULES = {
    "rule": "inv_var",        # the primary timing rule: size = c_t / σ̂², the papers' baseline
    "max_size": 1.5,          # 0 <= size <= 1.5 in the primary test; Phase 8 tries 1.0 and 2.0
    "min_months": 120,        # c_t needs 120 (signal, next-month return) pairs ...
    "etf_min_months": 36,     # ... 36 on the ETF track (Phase 7's three-year rule; MTUM starts in 2013).
                              #   c_t only sets the average size, and the Sharpe ratio ignores size
    "c_window": None,         # c_t uses every month up to t (None), not a rolling window
    "delay": 0,               # trade at the close of the month's last trading day, as the papers do
    "target_vol": 0.12,       # Barroso and Santa-Clara's fixed annual target: only to replicate them
    "placebo_draws": 1000,    # shuffled forecasts ...
    "placebo_share": 0.95,    # ... and the real alpha must beat 95% of their alphas
}
# Timing rule -> (forecaster, power): size = c_t x forecast^(-power). If volatility doubles, the size
# falls to 1/4 (power 1), 1/2 (power 0.5), or reacts only to bad days (downside variance).
TIMING = {
    "inv_var": (forecast.last_rv, 1.0),          # Moreira and Muir (2017)
    "inv_vol": (forecast.last_rv, 0.5),          # Barroso and Santa-Clara (2015)
    "down_vol": (forecast.last_rv_down, 0.5),    # Wang and Yan (2021)
}


# ---------------------------------------------------------------- the pipeline

def prepare(book: Book, forecaster) -> dict[int, pd.DataFrame]:
    """Stages 1 and 2. For each distinct recipe: the monthly table of the mix it holds, over all
    history, with the forecast made at each month-end."""
    ids = data.recipe_ids(book.positions)
    out = {}
    for k in ids.unique():
        pos = book.positions.loc[ids == k].iloc[0]
        p = data.mix(book, pos)
        out[k] = data.monthly(p, book.rf).join(forecaster(p).rename("forecast"), how="left")
    return out


def decide(book: Book, prepared: dict[int, pd.DataFrame], power: float, c="realtime",
           max_size: float | None = ENGINE_RULES["max_size"], min_months: int = ENGINE_RULES["min_months"],
           c_window: int | None = ENGINE_RULES["c_window"], seed: int | None = None) -> pd.DataFrame:
    """Stage 3. One row per holding month m: the forecast, signal, c and size, all made at the end
    of month m-1 from the recipe held in m. seed: shuffle each recipe's forecast in time (placebo).
    Months before the first size are dropped; a missing size after it is an error."""
    ids = data.recipe_ids(book.positions)
    rows = []
    for k, mon in prepared.items():
        f = mon["forecast"] if seed is None else forecast.permute(mon["forecast"], seed)
        s = weights.signal(f, power)
        ck = weights.realtime_c(mon["ret"], s, min_months, c_window) if c == "realtime" else pd.Series(float(c), s.index)
        held = ids.index[ids == k]
        t = held - 1                                            # decided at the end of the month before
        rows.append(pd.DataFrame({"recipe": k, "forecast": f.reindex(t).to_numpy(), "signal": s.reindex(t).to_numpy(),
                                  "c": ck.reindex(t).to_numpy()}, index=held))
    dec = pd.concat(rows).sort_index()
    dec["size"] = weights.size(dec["c"], dec["signal"], max_size).reindex(dec.index)
    dec["capped"] = dec["c"] * dec["signal"] > (np.inf if max_size is None else max_size)
    valid = dec["size"].notna()
    if not valid.any():
        raise ValueError("no month has a size: too little history for c_t")
    dec = dec.loc[valid.idxmax():]
    if dec["size"].isna().any():
        raise ValueError(f"no size for holding month {dec.index[dec['size'].isna()][0]}")
    return dec


def trade(book: Book, dec: pd.DataFrame, periods: execution.Periods) -> pd.DataFrame:
    """Stage 4, for the timed book (size x positions) and the buy-and-hold book (positions)."""
    pos = book.positions.loc[dec.index]
    timed = execution.simulate(periods, pos.mul(dec["size"], axis=0))
    bench = execution.simulate(periods, pos).add_prefix("bench_").drop(columns="bench_rf")
    return pd.concat([dec, timed, bench], axis=1)


def run(book: Book, rule: str = ENGINE_RULES["rule"], *, forecaster=None, c="realtime",
        max_size: float | None = ENGINE_RULES["max_size"], costs: Costs = NO_COSTS,
        delay: int = ENGINE_RULES["delay"], min_months: int = ENGINE_RULES["min_months"],
        c_window: int | None = ENGINE_RULES["c_window"]) -> pd.DataFrame:
    """Backtest one timing rule. One row per holding month, from the first month with a size.

    rule: a key of TIMING; forecaster replaces the rule's default forecast (same power).
    c: "realtime" (c_t from the months up to t) or a number (fixed, e.g. weights.target_c(0.12)).
    Columns: the decision (recipe, forecast, signal, c, size, capped), the timed account (gross,
    financing, pre_trade, turnover, trading, net, rf, excess, w_<i>, drift_<i>) and the same for
    buy-and-hold with the prefix bench_."""
    default, power = TIMING[rule]
    prepared = prepare(book, forecaster or default)
    dec = decide(book, prepared, power, c, max_size, min_months, c_window)
    return trade(book, dec, execution.holding_periods(book, costs, delay))


def placebo(book: Book, rule: str = ENGINE_RULES["rule"], *, forecaster=None,
            draws: int = ENGINE_RULES["placebo_draws"], seed: int = 0, **opts) -> dict:
    """The shuffled-forecast permutation test. Each draw reorders the forecast randomly in time and
    reruns weights and execution (c_t included). Returns the real alpha, the shuffled alphas, the
    share of them the real alpha beats, and pass (that share >= 0.95)."""
    default, power = TIMING[rule]
    periods = execution.holding_periods(book, opts.pop("costs", NO_COSTS), opts.pop("delay", ENGINE_RULES["delay"]))
    prepared = prepare(book, forecaster or default)
    real = trade(book, decide(book, prepared, power, **opts), periods)
    bench = real["bench_excess"]                               # buy-and-hold does not depend on the forecast

    def one(s):
        dec = decide(book, prepared, power, seed=s, **opts)
        timed = execution.simulate(periods, book.positions.loc[dec.index].mul(dec["size"], axis=0))
        return evaluation.alpha(timed["excess"], bench.reindex(dec.index))

    real = evaluation.alpha(real["excess"], bench)
    shuffled = np.array([one(seed + i) for i in range(draws)])
    share = evaluation.placebo_share(real, shuffled)
    return {"alpha": real, "shuffled": shuffled, "share_beaten": share,
            "pass": bool(share >= ENGINE_RULES["placebo_share"])}
