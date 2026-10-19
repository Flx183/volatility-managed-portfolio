"""
Weekly forecasts (Phase 7): the eight models at a 5-day horizon
===============================================================
A weekly variant rebalances at the close of each week's last trading day (weeks end on Friday), so
every model forecasts Σ r² over the next 5 trading days from data up to that close. The models, their
inputs and their refit schedule are Phase 6's (inputs.py, PREREGISTRATION Entries 4 and 6):

    M0          Σ r² over the last 5 days
    M1          5 x the EWMA one-step variance
    M2, M3      the parameters of Phase 6's month-end refits (the same cached fits). At each weekly
                origin σ² is filtered up to that day and the h-step forecasts are summed over 5 days
    M4, M5, M6  the HAR regressions on the 5-day target, refit at each month-end with a 5-day embargo.
                M6's event counts look 5 days ahead
    M7          boosting on the 5-day target, refit at its first forecast and each December, tuned by the
                same inner walk-forward with a 5-day embargo. Rows every fifth day, so the training
                targets do not overlap

Every weekly origin uses the latest fit made at a month-end on or before it, so a fit never sees data
after the origin. The first weekly forecast is at the first week ending on or after the mix's first
monthly origin (the end of its 120th month), so weekly and monthly forecasts start together.

This lives in its own module so that the Phase 6 forecast cache, keyed on each model file's code,
stays valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from sklearn.ensemble import HistGradientBoostingRegressor

from src.config import EVENTS
from src.engine.data import WEEK
from src.models import boost, filters, inputs
from src.models.inputs import entered, forward, har_design, qlike

H = 5                  # trading days: one holding week


# ---------------------------------------------------------------- when

def origins(p: pd.Series) -> pd.Series:
    """Position in p of the last trading day of each week (weeks end on Friday), by week."""
    p = p.dropna()
    return pd.Series(np.arange(len(p)), index=p.index.to_period(WEEK)).groupby(level=0).max()


def schedule(p: pd.Series, every: str = "month") -> tuple[pd.Series, np.ndarray]:
    """The weekly origins with a forecast, and for each the position of the fit it uses: the latest of
    Phase 6's refit origins on or before it (every month-end, or "year": the first and each December)."""
    p = p.dropna()
    mo, refit = inputs.schedule(p, every)
    fits = mo.to_numpy()[refit]
    wo = origins(p)
    wo = wo[wo >= mo.iloc[0]]
    return wo, fits[np.searchsorted(fits, wo.to_numpy(), side="right") - 1]


def at_origins(daily, p: pd.Series) -> pd.Series:
    """A daily series read at the weekly origins, by week."""
    wo, _ = schedule(p)
    return pd.Series(np.asarray(daily, dtype=float)[wo.to_numpy()], index=wo.index, name="forecast")


# ---------------------------------------------------------------- M0, M1

def rolling(p: pd.Series) -> pd.Series:
    p = p.dropna()
    return at_origins((p ** 2).rolling(H).sum(), p)


def ewma(p: pd.Series, lam: float = filters.LAMBDA) -> pd.Series:
    p = p.dropna()
    r2 = (p ** 2).to_numpy()
    s2_next = lfilter([1 - lam], [1, -lam], r2) / (1 - lam ** np.arange(1, len(r2) + 1))
    return at_origins(H * s2_next, p)


# ---------------------------------------------------------------- M2, M3

def garch(p: pd.Series, gjr: bool, params: pd.DataFrame, h: int = H) -> pd.Series:
    """M2 or M3 at weekly origins, from the parameters of the month-end refits (garch_fits' second
    output). With h = 22 and an origin on a month-end, this is exactly the monthly forecast."""
    p = p.dropna()
    r = p.to_numpy() * filters.SCALE
    r2 = r ** 2
    neg = np.where(r < 0, r2, 0.0)
    wo, fit_at = schedule(p)
    mo, _ = inputs.schedule(p)
    theta_at = dict(zip(mo.to_numpy(), params[["omega", "a", "g", "b"]].to_numpy()))
    d_all, out = wo.to_numpy(), np.empty(len(wo))
    for R in np.unique(fit_at):
        sel = fit_at == R
        d, theta = d_all[sel], theta_at[R]
        s2 = filters.garch_path(theta, r2[:d.max() + 1], neg[:d.max() + 1], r2[:R + 1].mean())   # started as in the fit
        w, a, g, b = theta
        s2_next = w + a * r2[d] + g * neg[d] + b * s2[d]
        out[sel] = filters.garch_sum(theta, s2_next, h) / filters.SCALE ** 2
    return pd.Series(out, index=wo.index, name="forecast")


# ---------------------------------------------------------------- M4, M5, M6

def exog(panel: pd.DataFrame, index: pd.DatetimeIndex, h: int = H) -> pd.DataFrame:
    """inputs.exog with the event counts over the next h trading days instead of 22."""
    x = panel.reindex(index)
    out = pd.DataFrame({"vix": np.log((x["vix"] / 100) ** 2 / 12)}, index=index)
    for e in EVENTS:
        flag = x[e].astype(float)
        count = flag.fillna(0.0)[::-1].rolling(h, min_periods=1).sum()[::-1].shift(-1, fill_value=0.0)   # d+1..d+h
        start = index.get_loc(flag.first_valid_index())
        count.iloc[:max(start - 1, 0)] = np.nan
        out[e] = count
    return out


def har_fits(p: pd.Series, down: bool = False, x: pd.DataFrame | None = None, h: int = H) -> tuple[pd.Series, pd.DataFrame]:
    """M4 (default), M5 (down=True) or M6 (x = exogenous columns) on the h-day target, refit at each
    month-end on the rows whose target is complete by then. The forecasts and each refit's fit."""
    p = p.dropna()
    X = har_design(p, down)
    extra = []
    if x is not None:
        X = X.join(x.reindex(p.index))
        extra = list(x.columns)
    base = [c for c in X.columns if c not in extra]
    A = X.to_numpy()
    y = np.log(forward(p ** 2, h)).to_numpy()
    wo, fit_at = schedule(p)
    d_all, out, fits = wo.to_numpy(), np.empty(len(wo)), []
    for R in np.unique(fit_at):
        last = R - h                                          # the last row whose target ends by R
        cols = base + entered(X, extra, last)
        idx = [X.columns.get_loc(c) for c in cols]
        Z, yy = A[:last + 1, idx], y[:last + 1]
        ok = np.isfinite(Z).all(axis=1) & np.isfinite(yy)
        Z1 = np.column_stack([np.ones(ok.sum()), Z[ok]])
        beta = np.linalg.lstsq(Z1, yy[ok], rcond=None)[0]
        resid = yy[ok] - Z1 @ beta
        s2 = resid @ resid / (len(resid) - Z1.shape[1])
        sel = fit_at == R
        out[sel] = np.exp(beta[0] + A[d_all[sel]][:, idx] @ beta[1:] + s2 / 2)
        fits.append({"fit_day": p.index[R], "const": beta[0], **dict(zip(cols, beta[1:])), "s2": s2, "rows": int(ok.sum())})
    return pd.Series(out, index=wo.index, name="forecast"), pd.DataFrame(fits).set_index("fit_day")


# ---------------------------------------------------------------- M7

def tune(A: np.ndarray, y: np.ndarray, rows: np.ndarray, dates: pd.DatetimeIndex, h: int) -> dict:
    """boost.tune with an h-day embargo: each of the last three years of training rows is forecast by a
    model trained on rows whose targets end before that year starts."""
    end = dates[rows[-1]]
    score = np.zeros((len(boost.LEAVES), len(boost.TREES)))
    for j in range(boost.FOLDS, 0, -1):
        lo, hi = end - pd.DateOffset(years=j), end - pd.DateOffset(years=j - 1)
        val = rows[(dates[rows] > lo) & (dates[rows] <= hi)]
        train = rows[rows <= val[0] - h]
        for i, leaves in enumerate(boost.LEAVES):
            m = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=max(boost.TREES), **boost.SETTINGS)
            m.fit(A[train], y[train])
            for k, pred in enumerate(m.staged_predict(A[val]), start=1):
                if k in boost.TREES:
                    score[i, boost.TREES.index(k)] += qlike(y[val], pred).mean() / boost.FOLDS
    i, k = np.unravel_index(np.argmin(score), score.shape)
    return {"leaves": boost.LEAVES[i], "trees": boost.TREES[k], "inner_qlike": score[i, k]}


def boost_fits(p: pd.Series, down: bool, x: pd.DataFrame, h: int = H) -> tuple[pd.Series, pd.DataFrame]:
    """M7 on the h-day target, refit at its first forecast and each December. The forecasts and each
    refit's tuned setting."""
    p = p.dropna()
    X = har_design(p, down).join(x.reindex(p.index))
    extra = list(x.columns)
    base_idx = [X.columns.get_loc(c) for c in X.columns if c not in extra]
    A, y, dates = X.to_numpy(), forward(p ** 2, h).to_numpy(), p.index
    wo, fit_at = schedule(p, "year")
    d_all, out, fits = wo.to_numpy(), np.empty(len(wo)), []
    for R in np.unique(fit_at):
        last = R - h
        cols = [c for c in X.columns if c not in extra] + entered(X, extra, last)
        idx = [X.columns.get_loc(c) for c in cols]
        rows = np.arange(last, -1, -boost.EVERY)[::-1]        # every fifth day, ending at the last usable row
        rows = rows[np.isfinite(y[rows]) & np.isfinite(A[rows][:, base_idx]).all(axis=1)]
        best = tune(A[:, idx], y, rows, dates, h)
        model = HistGradientBoostingRegressor(max_leaf_nodes=best["leaves"], max_iter=best["trees"], **boost.SETTINGS)
        model.fit(A[rows][:, idx], y[rows])
        sel = fit_at == R
        out[sel] = model.predict(A[d_all[sel]][:, idx])
        fits.append({"fit_day": p.index[R], **best, "rows": len(rows), "inputs": " ".join(cols)})
    return pd.Series(out, index=wo.index, name="forecast"), pd.DataFrame(fits).set_index("fit_day")
