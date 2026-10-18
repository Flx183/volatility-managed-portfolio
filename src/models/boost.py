"""
M7: gradient boosting on M6's inputs
====================================
The same inputs as M6 (the HAR lags, plus the down-day shares if M6 uses them, VIX and event
counts), but the forecast can bend: trees can find thresholds and interactions a regression cannot.

    loss       gamma deviance, 2 x (y/ŷ - ln(y/ŷ) - 1) = 2 x QLIKE. Fitted on the 22-day variance itself
               (log link), so it forecasts E[RV] directly and needs no back-transform
    rows       every fifth trading day: neighbouring 22-day targets overlap by 21 days, so the other
               rows add little, and boosting is the slow model
    refit      at the first forecast and each December (yearly, for compute), on all usable rows
    tuning     at each refit, an inner walk-forward inside the training rows only: each of the last
               three years is forecast by a model trained on rows whose targets end before that year
               starts (the 22-day embargo), and the setting with the lowest mean QLIKE wins
    grid       max_leaf_nodes 4, 8, 16 x trees 25, 50, 100, 200, 400; learning rate 0.05,
               at least 50 rows a leaf, no early stopping (its random split would leak across
               overlapping targets)

VIX and event counts enter at a refit once they have 60 months of history, as in M6. Boosting
handles missing values, so the earlier rows stay in the training set with those inputs missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.inputs import H, entered, forward, har_design, qlike, schedule

LEAVES = (4, 8, 16)
TREES = (25, 50, 100, 200, 400)
SETTINGS = {"loss": "gamma", "learning_rate": 0.05, "min_samples_leaf": 50, "early_stopping": False,
            "random_state": 0}
EVERY = 5
FOLDS = 3


def boost(p: pd.Series, down: bool, x: pd.DataFrame) -> pd.Series:
    """M7: the forecast only."""
    return boost_fits(p, down, x)[0]


def tune(A: np.ndarray, y: np.ndarray, rows: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    """The inner walk-forward. Returns the winning setting and its mean QLIKE over the three folds."""
    end = dates[rows[-1]]
    score = np.zeros((len(LEAVES), len(TREES)))
    for j in range(FOLDS, 0, -1):
        lo, hi = end - pd.DateOffset(years=j), end - pd.DateOffset(years=j - 1)
        val = rows[(dates[rows] > lo) & (dates[rows] <= hi)]
        train = rows[rows <= val[0] - H]                   # targets end by the fold's first day
        for i, leaves in enumerate(LEAVES):
            m = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=max(TREES), **SETTINGS)
            m.fit(A[train], y[train])
            for k, pred in enumerate(m.staged_predict(A[val]), start=1):
                if k in TREES:
                    score[i, TREES.index(k)] += qlike(y[val], pred).mean() / FOLDS
    i, k = np.unravel_index(np.argmin(score), score.shape)
    return {"leaves": LEAVES[i], "trees": TREES[k], "inner_qlike": score[i, k]}


def boost_fits(p: pd.Series, down: bool, x: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """The forecasts and, for each refit, the tuned setting, its inner QLIKE, the rows and the inputs used."""
    p = p.dropna()
    X = har_design(p, down).join(x.reindex(p.index))
    extra = list(x.columns)
    base_idx = [X.columns.get_loc(c) for c in X.columns if c not in extra]
    A, y, dates = X.to_numpy(), forward(p ** 2).to_numpy(), p.index
    o, refit = schedule(p, "year")
    out, fits, model, idx = [], [], None, None
    for month, pos, re in zip(o.index, o.to_numpy(), refit):
        if re:
            last = pos - H
            cols = [c for c in X.columns if c not in extra] + entered(X, extra, last)
            idx = [X.columns.get_loc(c) for c in cols]
            rows = np.arange(last, -1, -EVERY)[::-1]       # every fifth day, ending at the last usable row
            rows = rows[np.isfinite(y[rows]) & np.isfinite(A[rows][:, base_idx]).all(axis=1)]
            best = tune(A[:, idx], y, rows, dates)
            model = HistGradientBoostingRegressor(max_leaf_nodes=best["leaves"], max_iter=best["trees"], **SETTINGS)
            model.fit(A[rows][:, idx], y[rows])
            fits.append({"month": month, **best, "rows": len(rows), "inputs": " ".join(cols)})
        out.append(model.predict(A[[pos]][:, idx])[0])
    return pd.Series(out, index=o.index, name="forecast"), pd.DataFrame(fits).set_index("month")
