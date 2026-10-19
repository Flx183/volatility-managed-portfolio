"""
The GARCH forecasts under Phase 8's variations
==============================================
Two robustness checks change how the frozen models (M2, M3) forecast, and nothing else
(PREREGISTRATION Entry 8):

    garch_daily(p, gjr, params)          rebalancing every trading day: at each close, tomorrow's
                                         variance σ²_{d+1} (a one-day horizon), from the latest
                                         month-end fit, with σ² filtered up to that close
    garch_fits_window(p, gjr, days)      a rolling estimation window: each month-end refit uses only
                                         the last `days` trading days, and the forecast filters σ²
                                         over the same window

Everything else is Phase 6's (filters.py): zero-mean Gaussian QMLE on returns in percent, the
recursion started at the mean r² of the estimation data, each refit started from the last estimate
and never ending at a worse likelihood (Entry 5), the first forecast at the end of the mix's 120th
month. With days >= the data's length, garch_fits_window is Phase 6's expanding walk-forward.

This lives in its own module so the forecast cache of Phases 6 and 7, keyed on each model file's
code, stays valid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.data import DAY
from src.models import filters, inputs


def daily_schedule(p: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Every trading day from the mix's first monthly origin (the end of its 120th month), by day, and
    for each the position of the fit it uses: the latest month-end refit on or before it."""
    p = p.dropna()
    mo, _ = inputs.schedule(p)
    fits = mo.to_numpy()
    pos = np.arange(fits[0], len(p))
    days = pd.Series(pos, index=p.index[pos].to_period(DAY))
    return days, fits[np.searchsorted(fits, pos, side="right") - 1]


def garch_daily(p: pd.Series, gjr: bool, params: pd.DataFrame) -> pd.Series:
    """M2 or M3 one trading day ahead, at every close: σ²_{d+1} = ω + a r²_d + g r²_d 1{r_d < 0} + b σ²_d
    with the parameters of the latest month-end refit (garch_fits' second output), the path started as
    in that fit. On a month-end this is the first term of the monthly forecast's sum."""
    p = p.dropna()
    r = p.to_numpy() * filters.SCALE
    r2 = r ** 2
    neg = np.where(r < 0, r2, 0.0)
    days, fit_at = daily_schedule(p)
    mo, _ = inputs.schedule(p)
    theta_at = dict(zip(mo.to_numpy(), params[["omega", "a", "g", "b"]].to_numpy()))
    d_all, out = days.to_numpy(), np.empty(len(days))
    for R in np.unique(fit_at):
        sel = fit_at == R
        d, theta = d_all[sel], theta_at[R]
        s2 = filters.garch_path(theta, r2[:d.max() + 1], neg[:d.max() + 1], r2[:R + 1].mean())
        w, a, g, b = theta
        out[sel] = (w + a * r2[d] + g * neg[d] + b * s2[d]) / filters.SCALE ** 2
    return pd.Series(out, index=days.index, name="forecast")


def garch_fits_window(p: pd.Series, gjr: bool, days: int) -> tuple[pd.Series, pd.DataFrame]:
    """Phase 6's monthly walk-forward (garch_fits) on a rolling window: at each month-end the fit and
    the filtered σ² use only the last `days` trading days up to that close. The forecasts and the
    parameters of each refit."""
    p = p.dropna()
    r = p.to_numpy() * filters.SCALE
    r2 = r ** 2
    neg = np.where(r < 0, r2, 0.0)
    o, _ = inputs.schedule(p)
    out, params, theta = [], [], None
    for month, pos in zip(o.index, o.to_numpy()):
        lo, n = max(0, pos + 1 - days), pos + 1                 # the window: the last `days` days to the close
        x2, ng = r2[lo:n], neg[lo:n]
        init = x2.mean()
        theta, ok = filters.garch_fit(x2, ng, init, gjr, theta)
        params.append({"month": month, "omega": theta[0], "a": theta[1], "g": theta[2], "b": theta[3], "converged": ok,
                       "window_start": p.index[lo]})
        s2 = filters.garch_path(theta, x2, ng, init)[-1]
        w, a, g, b = theta
        out.append(filters.garch_sum(theta, w + a * x2[-1] + g * ng[-1] + b * s2) / filters.SCALE ** 2)
    return pd.Series(out, index=o.index, name="forecast"), pd.DataFrame(params).set_index("month")
