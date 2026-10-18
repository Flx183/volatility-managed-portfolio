"""
M4-M6: HAR regressions in logs
==============================
    M4 har        log RV_{t+1:t+22} = β0 + β_d log RV_d + β_w log RV_w + β_m log RV_m + e
    M5 har_down   M4 plus the down-day share of each lag (inputs.har_design, down=True)
    M6 har_x      M4 or M5 plus log VIX variance and the counts of FOMC, CPI and jobs days ahead

Direct: each model is fitted on the 22-day target itself, on every trading day (overlapping targets),
so nothing is iterated forward. Refit at every month-end on all usable rows before it (expanding
window, with the 22-day embargo of inputs.py), by ordinary least squares.

Back to variance with a correction. If log RV is normal with mean μ and variance s², then
E[RV] = exp(μ + s²/2), not exp(μ). So the forecast is exp(fitted log RV + s²/2), with s² the residual
variance of the fit. Without it the forecasts run low; the strategy would absorb a constant scale
error into c_t, but QLIKE punishes it.

M6's extra columns enter a refit once they have 60 months of history (VIX from 1995, events from
1999); the regression is then fitted on the rows that have them. Before that, M6 is its base model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.inputs import H, entered, forward, har_design, schedule


def har(p: pd.Series, down: bool = False, x: pd.DataFrame | None = None) -> pd.Series:
    """M4 (default), M5 (down=True) or M6 (x = exogenous columns): the forecast only."""
    return har_fits(p, down, x)[0]


def har_fits(p: pd.Series, down: bool = False, x: pd.DataFrame | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """The forecasts and, for each refit, the coefficients, the residual variance s² and the rows used."""
    p = p.dropna()
    X = har_design(p, down)
    extra = []
    if x is not None:
        X = X.join(x.reindex(p.index))
        extra = list(x.columns)
    base = [c for c in X.columns if c not in extra]
    A = X.to_numpy()
    y = np.log(forward(p ** 2)).to_numpy()
    o, _ = schedule(p)
    out, fits = [], []
    for month, pos in o.items():
        last = pos - H                                        # the last row whose target ends by the origin
        cols = base + entered(X, extra, last)
        idx = [X.columns.get_loc(c) for c in cols]
        Z, yy = A[:last + 1, idx], y[:last + 1]
        ok = np.isfinite(Z).all(axis=1) & np.isfinite(yy)
        Z1 = np.column_stack([np.ones(ok.sum()), Z[ok]])
        beta = np.linalg.lstsq(Z1, yy[ok], rcond=None)[0]
        resid = yy[ok] - Z1 @ beta
        s2 = resid @ resid / (len(resid) - Z1.shape[1])
        out.append(np.exp(beta[0] + A[pos, idx] @ beta[1:] + s2 / 2))
        fits.append({"month": month, "const": beta[0], **dict(zip(cols, beta[1:])), "s2": s2, "rows": int(ok.sum())})
    return pd.Series(out, index=o.index, name="forecast"), pd.DataFrame(fits).set_index("month")
