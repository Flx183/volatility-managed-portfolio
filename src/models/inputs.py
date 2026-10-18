"""
Phase 6 inputs: what every model sees
=====================================
Every model forecasts the same quantity. At the close of month t's last trading day (the origin), it
forecasts the total variance of the timed mix over the next 22 trading days, Σ r² over days t+1..t+22,
from data up to that close. 22 trading days is one holding month, matching monthly rebalancing. A weekly
rule would use 5.

    origins(p)          position in p of each month's last trading day, by month (the engine's months)
    schedule(p, every)  the origins that get a forecast (from the mix's 120th month) and which ones refit
    forward(x)          Σ x over the next 22 days, at each day: the direct target
    target(p)           the realized 22-day variance after each origin (what Scorecard A scores against)
    har_design(p)       HAR regressors: log of the 1-, 5- and 22-day mean r², and the down-day share of each
    exog(panel, index)  VIX as a log monthly variance, and scheduled-event counts in the next 22 days
    entered(...)        which extra columns have enough history to enter a refit

Training rows and the embargo. A training row at day d has the target Σ r² over d+1..d+22. At a
refit on origin day R, a row is usable only if its target is complete by R, so d <= R - 22. The last
22 rows are dropped: their targets overlap the forecast period, so keeping them would leak the future.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import variance as v
from src.config import EVENTS, PROCESSED_DIR

H = 22                 # forecast horizon, trading days: one holding month
MIN_MONTHS = 120       # the first forecast is made at the end of the mix's 120th month, by every model
FLOOR = 1e-3           # a day's r² is floored at 1/1000 of its 22-day mean before taking logs
X_MIN_MONTHS = 60      # VIX and event counts enter a refit once they have 60 months of history

CACHE_ROOT: Path = PROCESSED_DIR / "phase6_cache"   # forecasts cached on disk (src/models/__init__.py)


def origins(p: pd.Series) -> pd.Series:
    """Position in p of the last trading day of each month the engine keeps (>= 10 trading days)."""
    p = p.dropna()
    last = pd.Series(np.arange(len(p)), index=p.index.to_period("M")).groupby(level=0).max()
    return last.loc[v.monthly_table(p).index]


def schedule(p: pd.Series, every: str = "month") -> tuple[pd.Series, np.ndarray]:
    """Origins with a forecast (from the end of the mix's 120th month) and whether each one refits:
    every origin ("month"), or the first and each December ("year")."""
    o = origins(p).iloc[MIN_MONTHS - 1:]
    if every == "month":
        return o, np.ones(len(o), dtype=bool)
    return o, (o.index.month == 12) | (np.arange(len(o)) == 0)


def forward(x: pd.Series, h: int = H) -> pd.Series:
    """Σ x over days d+1..d+h, at each day d (NaN where the window runs past the data)."""
    return x.rolling(h).sum().shift(-h)


def at_origins(daily: np.ndarray | pd.Series, p: pd.Series, every: str = "month") -> pd.Series:
    """A daily series read at the forecast origins, by month."""
    o, _ = schedule(p, every)
    return pd.Series(np.asarray(daily, dtype=float)[o.to_numpy()], index=o.index, name="forecast")


def target(p: pd.Series) -> pd.Series:
    """Realized variance over the 22 trading days after each origin (every month, not only forecast ones)."""
    p = p.dropna()
    o = origins(p)
    return pd.Series(forward(p ** 2).to_numpy()[o.to_numpy()], index=o.index, name="target")


def har_design(p: pd.Series, down: bool = False) -> pd.DataFrame:
    """Daily HAR regressors (Phase 2's Test 2, in logs): d, w, m = log of the mean r² over the last
    1, 5 and 22 days. The 1- and 5-day terms are floored at 1/1000 of the 22-day mean, because a day
    with a near-zero return has log r² near minus infinity (about 2.6% of days, as normal theory predicts).

    down=True adds the down-day share of each window, Σ r² on down days / Σ r²: for the 1-day lag that
    is 1 on a down day and 0 on an up day. Splitting each lag into log downside and log upside variance
    fails in logs, because one of the two is zero on every single day. The shares carry the same split:
    log RV + δ x share lets a unit of down-day variance predict more than a unit of up-day variance."""
    p = p.dropna()
    r2 = p ** 2
    m = r2.rolling(22).mean()
    X = pd.DataFrame({"d": np.log(np.maximum(r2, FLOOR * m)),
                      "w": np.log(np.maximum(r2.rolling(5).mean(), FLOOR * m)),
                      "m": np.log(m)}, index=p.index)
    if down:
        dn = r2.where(p < 0, 0.0)
        for k, n in (("d", 1), ("w", 5), ("m", 22)):
            total = r2.rolling(n).sum()
            X[f"{k}_down"] = (dn.rolling(n).sum() / total).where(total > 0, 0.5)
    return X


def exog(panel: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """VIX as a log monthly variance, log((VIX/100)² / 12), and the number of scheduled FOMC, CPI and
    jobs-report days in the next 22 trading days. The events are on the calendar in advance (Phase 1
    keeps only scheduled ones), so their count is known at the origin. NaN where the data do not reach:
    VIX before 1990, events before 1994. Days after the development sample count as no event."""
    x = panel.reindex(index)
    out = pd.DataFrame({"vix": np.log((x["vix"] / 100) ** 2 / 12)}, index=index)
    for e in EVENTS:
        flag = x[e].astype(float)
        count = flag.fillna(0.0)[::-1].rolling(H, min_periods=1).sum()[::-1].shift(-1, fill_value=0.0)   # d+1..d+22
        start = index.get_loc(flag.first_valid_index())
        count.iloc[:max(start - 1, 0)] = np.nan          # windows that begin before the calendar starts
        out[e] = count
    return out


def qlike(rv, forecast):
    """QLIKE = RV/σ̂² - ln(RV/σ̂²) - 1: zero for a perfect forecast; under-forecasting costs more.
    Scorecard A's loss, and the loss M7 is tuned on."""
    ratio = np.asarray(rv, dtype=float) / np.asarray(forecast, dtype=float)
    return ratio - np.log(ratio) - 1


def entered(X: pd.DataFrame, extra: list[str], last_row: int) -> list[str]:
    """The extra columns with at least X_MIN_MONTHS of history up to training row last_row."""
    end = X.index[last_row]
    out = []
    for c in extra:
        first = X[c].first_valid_index()
        if first is not None and first + pd.DateOffset(months=X_MIN_MONTHS) <= end:
            out.append(c)
    return out
