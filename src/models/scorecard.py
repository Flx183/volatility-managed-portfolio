"""
Phase 6 scorecards and the selection rule
=========================================
    books(panel)            the two books scored: the primary (paper track, inverse-volatility mix)
                            and the SPY reference (the market factor, traded at SPY's costs)
    run(book, f, costs)     the engine at the primary specification, with forecaster f
    targets(book)           per holding month, the realized 22-day variance after its decision, for
                            the mix it holds: what Scorecard A scores the forecast against
    mcs(losses)             the Model Confidence Set (Hansen, Lunde and Nason 2011)
    scorecard_a(runs, y)    forecast accuracy: mean QLIKE, calibration, MCS p-value
    scorecard_b(runs)       strategy value: Sharpe ratio, CER and realized volatility, and the rest
    select(a, runs)         the rule fixed in advance: the simplest model inside the MCS whose Sharpe
                            ratio is within one standard error of the best
    rank_agreement(a, b)    Spearman correlation of the two rankings (question Q4)

Why two scorecards. The strategy cares only about how σ̂² moves, because c_t absorbs its level, so a
forecast can win on QLIKE and lose on Sharpe. If the rankings barely agree, forecast accuracy is not
the bottleneck.

Rules in SCORE_RULES (PREREGISTRATION Entry 4), fixed before any real-data score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import engine as e
from src import portfolio as pf
from src.engine import evaluation as ev
from src.models import ORDER
from src.models.inputs import qlike, target

SCORE_RULES = {
    "rule": "inv_var",      # the primary specification: inverse-variance timing ...
    "max_size": 1.5,        # ... capped at 1.5, monthly, costs of each track (Entry 2), real-time c_t
    "mcs_level": 0.10,      # the MCS keeps every model with an MCS p-value of at least 10% (a 90% set)
    "mcs_block": 12,        # stationary bootstrap, mean block of 12 months (losses cluster in crises)
    "mcs_reps": 10_000,
    "seed": 0,
    "gamma": 5,             # risk aversion of the CER
}
# The SPY reference is chosen on the market factor, SPY's long-history stand-in: ETFs are never tuned
# on. It pays SPY's costs (Entry 2): 5 bp a trade, borrowed cash at rf + 0.5% a year.
SPY_COSTS = e.Costs(kappa={"mkt": 5e-4}, borrow_spread=0.005)


def books(panel: pd.DataFrame) -> dict[str, dict]:
    """book, costs, and the daily series the Phase 2 hints are re-run on, for each scored book."""
    rec = pf.recipes(panel, "paper")
    method = pf.PRIMARY[1]
    return {
        "paper": {"book": e.data.from_phase4(panel, rec, method), "costs": e.COSTS["paper"],
                  "series": pf.returns(panel, "paper", rec)[(method, "p")].dropna()},
        "spy": {"book": e.Book.single(panel["mkt"], panel["rf"], "mkt"), "costs": SPY_COSTS,
                "series": panel["mkt"].dropna()},
    }


def run(book: e.Book, forecaster, costs: e.Costs, **opts) -> pd.DataFrame:
    return e.run(book, SCORE_RULES["rule"], forecaster=forecaster, costs=costs,
                 max_size=SCORE_RULES["max_size"], **opts)


def targets(book: e.Book) -> pd.Series:
    """For each holding month m: Σ r² over the 22 trading days after month m-1's close, for the mix
    held in m. NaN where those days run past the development sample (the last month)."""
    ids = e.data.recipe_ids(book.positions)
    out = pd.Series(np.nan, index=book.positions.index, name="target")
    for k in ids.unique():
        held = ids.index[ids == k]
        y = target(e.data.mix(book, book.positions.loc[held[0]]))
        out.loc[held] = y.reindex(held - 1).to_numpy()
    return out


# ---------------------------------------------------------------- Scorecard A

def _bootstrap_means(L: np.ndarray, block: int, reps: int, rng) -> np.ndarray:
    """Means of each column of L over `reps` stationary-bootstrap resamples (Politis and Romano):
    blocks of random start and geometric length with mean `block`, wrapping around the end."""
    n = len(L)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    jump = rng.random((reps, n)) < 1 / block
    fresh = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    counts = np.bincount((np.arange(reps)[:, None] * n + idx).ravel(), minlength=reps * n).reshape(reps, n)
    return counts @ L / n


def mcs(losses: pd.DataFrame, level: float = SCORE_RULES["mcs_level"], block: int = SCORE_RULES["mcs_block"],
        reps: int = SCORE_RULES["mcs_reps"], seed: int = SCORE_RULES["seed"]) -> pd.DataFrame:
    """The Model Confidence Set with the T_max statistic. Each round compares every model's mean loss
    with the average of the models left: t_i = (mean L_i - mean over models) / its bootstrap standard
    error, and asks whether the largest t_i is too large to be chance (a bootstrap p-value). The model
    with the largest t_i is dropped and the round repeats, down to one model. A model's MCS p-value is
    the largest p-value of any round up to the one that dropped it (1 for the last model left); the
    set is every model whose MCS p-value is at least `level`. One set of bootstrap draws serves every
    round, so the eight models are compared jointly, not in 28 separate tests."""
    L = losses.to_numpy(dtype=float)
    boot = _bootstrap_means(L, block, reps, np.random.default_rng(seed))
    left, p_mcs, order, running = list(range(L.shape[1])), {}, {}, 0.0
    while len(left) > 1:
        mean = L[:, left].mean(axis=0)
        d = mean - mean.mean()
        d_star = boot[:, left] - boot[:, left].mean(axis=1, keepdims=True) - d
        se = np.sqrt((d_star ** 2).mean(axis=0))
        t = d / se
        running = max(running, float(((d_star / se).max(axis=1) >= t.max()).mean()))
        worst = left[int(np.argmax(t))]
        p_mcs[worst], order[worst] = running, len(order) + 1
        left.remove(worst)
    p_mcs[left[0]], order[left[0]] = 1.0, len(order) + 1
    res = pd.DataFrame({"mcs_p": p_mcs, "dropped_in_round": order}).sort_index()
    res.index = losses.columns
    res["in_mcs"] = res["mcs_p"] >= level
    return res


def losses(runs: dict[str, pd.DataFrame], y: pd.Series) -> pd.DataFrame:
    """QLIKE per holding month and model, on the months every model and the target cover."""
    index = _common_index(runs)
    f = pd.DataFrame({m: r["forecast"] for m, r in runs.items()}).reindex(index)
    y = y.reindex(index)
    ok = y.notna() & f.notna().all(axis=1)
    return pd.DataFrame(qlike(y[ok].to_numpy()[:, None], f[ok].to_numpy()), index=index[ok], columns=f.columns)


def scorecard_a(runs: dict[str, pd.DataFrame], y: pd.Series) -> pd.DataFrame:
    L = losses(runs, y)
    f = pd.DataFrame({m: r["forecast"] for m, r in runs.items()}).reindex(L.index)
    out = pd.DataFrame({"qlike": L.mean(), "qlike_median": L.median(),
                        "rv_over_forecast": y.reindex(L.index).mean() / f.mean(),
                        "months": len(L)})
    out = out.join(mcs(L))
    out["qlike_rank"] = out["qlike"].rank().astype(int)
    return out


# ---------------------------------------------------------------- Scorecard B

def _common_index(runs: dict[str, pd.DataFrame]) -> pd.PeriodIndex:
    index = next(iter(runs.values())).index
    for m, r in runs.items():
        if not r.index.equals(index):
            raise ValueError(f"{m} covers different holding months; every model must start at the same month")
    return index


def scorecard_b(runs: dict[str, pd.DataFrame], gamma: float = SCORE_RULES["gamma"]) -> pd.DataFrame:
    _common_index(runs)
    rows = {}
    for m, res in runs.items():
        test = ev.sharpe_gain_test(res)
        span = ev.spanning(res["excess"], res["bench_excess"])
        rows[m] = {"sharpe": ev.sharpe(res["excess"]), "cer": ev.cer(res["excess"], gamma),
                   "volatility": res["excess"].std() * np.sqrt(12), "mean": res["excess"].mean() * 12,
                   "gain": test["gain"], "gain_se": test["se"], "gain_p": test["p"], "reading": test["reading"],
                   "alpha": span["alpha"], "t_alpha": span["t_alpha"], "mean_size": res["size"].mean(),
                   "share_capped": res["capped"].mean(), "turnover_a_year": res["turnover"].iloc[1:].mean() * 12,
                   "break_even_kappa": ev.break_even(res), "months": len(res)}
    out = pd.DataFrame(rows).T
    out["sharpe_rank"] = out["sharpe"].astype(float).rank(ascending=False).astype(int)
    return out


def buy_and_hold(res: pd.DataFrame, gamma: float = SCORE_RULES["gamma"]) -> dict:
    """The benchmark row: the same book at size 1, on the same months (identical in every run)."""
    x = res["bench_excess"]
    return {"sharpe": ev.sharpe(x), "cer": ev.cer(x, gamma), "volatility": x.std() * np.sqrt(12),
            "mean": x.mean() * 12, "mean_size": 1.0, "turnover_a_year": res["bench_turnover"].iloc[1:].mean() * 12,
            "months": len(res)}


# ---------------------------------------------------------------- the selection rule

def select(a: pd.DataFrame, runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The simplest model (ORDER) inside the MCS whose Sharpe ratio is within one standard error of the
    best of the eight. The standard error is that of the gap Sharpe(best) - Sharpe(model), by the
    primary test's method (Ledoit-Wolf, Newey-West 6 lags): the eight runs share their months and most
    of their returns, so the gap's own error is what says whether two models tie. If no model of the
    MCS is within one error, the MCS condition is dropped: the simplest model within one error."""
    sharpe = {m: ev.sharpe(runs[m]["excess"]) for m in ORDER}
    best = max(sharpe, key=sharpe.get)
    rows = {}
    for m in ORDER:
        gap, se = (0.0, 0.0) if m == best else ev.sharpe_difference(runs[best]["excess"], runs[m]["excess"])
        rows[m] = {"sharpe": sharpe[m], "gap_to_best": gap, "gap_se": se, "within_1se": bool(gap <= se),
                   "in_mcs": bool(a.loc[m, "in_mcs"]), "best": m == best}
    t = pd.DataFrame(rows).T
    eligible = t["within_1se"] & t["in_mcs"]
    t["fallback"] = not eligible.any()
    pick = (eligible if eligible.any() else t["within_1se"]).astype(bool).idxmax()   # the first True in ORDER
    t["chosen"] = t.index == pick
    return t


def rank_agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Spearman correlation between the QLIKE ranking and the Sharpe ranking of the same models:
    +1 when the most accurate forecast is also the best strategy, near 0 when accuracy does not matter."""
    return float(a["qlike_rank"].corr(b.loc[a.index, "sharpe_rank"].astype(float), method="spearman"))
