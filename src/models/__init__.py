"""
Phase 6: variance forecast models
=================================
Eight forecasts of the same thing: at each month-end close, the timed mix's total variance over the
next 22 trading days (one holding month), from data up to that close. Simplest first:

    M0 rolling    Σ r² over the last 22 days                                 the papers' baseline
    M1 ewma       22 x EWMA of r², λ = 0.94                                  no fitting
    M2 garch      GARCH(1,1), h-step forecasts summed over 22 days           short memory
    M3 gjr        GJR-GARCH: GARCH plus an extra kick after down days        short memory, asymmetry
    M4 har        HAR in logs on 1-, 5-, 22-day log RV, direct 22-day target long memory
    M5 har_down   M4 plus each lag's down-day share                          long memory, asymmetry
    M6 har_x      M4 or M5 plus log VIX variance and scheduled-event counts  VIX adds information
    M7 boost      gradient boosting on M6's inputs, QLIKE (gamma) loss       nonlinearity

Each is an engine forecaster (src/engine/forecast.py): f(p) takes the mix's daily excess returns
and returns one forecast per month, made at its last trading day. So each plugs into
e.run(book, forecaster=f) and nothing else changes. Every model makes its first forecast at the end
of the mix's 120th month, so all eight are scored on the same months.

The protocol (inputs.py): expanding windows; refit at every month-end (M7: yearly, for compute);
direct models drop the last 22 days of training targets (the embargo); M7 is tuned by an inner
walk-forward inside its training window only. Rules are in MODEL_RULES (PREREGISTRATION Entry 4).

Development sample only. The slow forecasts are cached on disk (data/processed/phase6_cache/);
`python -m src.models.build` fills the cache in parallel.

Phase 7 adds weekly versions (weekly.py): the same eight models at a 5-day horizon, forecast at each
week's last trading day. forecasters(panel, down, freq="W-FRI") returns them.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd

from src import variance as v
from src.engine import forecast as engine_forecast
from src.engine.data import MONTH, WEEK
from src.models import boost, filters, har, inputs, weekly
from src.models.inputs import schedule

# ---------------------------------------------------------------- rules
# Fixed on 2026-10-18 before any real-data forecast was scored (PREREGISTRATION Entry 4).
MODEL_RULES = {
    "horizon": inputs.H,                  # trading days: one holding month
    "min_months": inputs.MIN_MONTHS,      # first forecast at the end of the mix's 120th month, every model
    "embargo": inputs.H,                  # direct models drop the last 22 days of training targets
    "refit": {"M0": None, "M1": None, "M2": "month", "M3": "month", "M4": "month", "M5": "month",
              "M6": "month", "M7": "year"},
    "ewma_lambda": filters.LAMBDA,
    "log_floor": inputs.FLOOR,            # a day's r² is floored at 1/1000 of its 22-day mean in logs
    "x_min_months": inputs.X_MIN_MONTHS,  # VIX and event counts enter once they have 60 months of history
    "m6_base": "M5 if Phase 2 Test 3 passes on the series forecast, else M4",
}
ORDER = ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]   # simplest first: the selection rule's order
NAMES = {"M0": "Rolling RV", "M1": "EWMA", "M2": "GARCH(1,1)", "M3": "GJR-GARCH", "M4": "HAR (log)",
         "M5": "Downside HAR", "M6": "HAR-X", "M7": "Gradient boosting"}

def _cached(fn, also=()):
    """fn, cached on disk by the hash of its inputs, in a folder named by the hash of the code it runs
    (its module, inputs.py, Phase 2's variance.py and any modules in `also`). A change to that code
    starts a fresh folder, so a stale forecast is never read, and a change to one model leaves the
    others' caches alone."""
    code = [Path(inspect.getfile(fn)), Path(inputs.__file__), Path(v.__file__), *(Path(m.__file__) for m in also)]
    folder = inputs.CACHE_ROOT / hashlib.sha256(b"".join(f.read_bytes() for f in code)).hexdigest()[:12]
    return joblib.Memory(folder, verbose=0).cache(fn)


garch_fits = _cached(filters.garch_fits)
har_fits = _cached(har.har_fits)
boost_fits = _cached(boost.boost_fits)
# Weekly (Phase 7): weekly.py also runs code from filters.py and boost.py, so their hashes are in the key.
garch_weekly = _cached(weekly.garch, also=(filters, boost))
har_weekly = _cached(weekly.har_fits, also=(filters, boost))
boost_weekly = _cached(weekly.boost_fits, also=(filters, boost))


def exog(panel: pd.DataFrame, p: pd.Series) -> pd.DataFrame:
    """M6 and M7's extra inputs for the mix p. Build them only this way: the cache key is a hash of
    the inputs, and an equal DataFrame built another way can hash differently."""
    return inputs.exog(panel, p.dropna().index)


def weekly_exog(panel: pd.DataFrame, p: pd.Series) -> pd.DataFrame:
    """exog() for the weekly models: event counts over the next 5 days. Build them only this way."""
    return weekly.exog(panel, p.dropna().index)


def forecasters(panel: pd.DataFrame, down: bool, freq: str = MONTH) -> dict[str, Callable[[pd.Series], pd.Series]]:
    """The eight models as engine forecasters, in ORDER. down: whether M6 and M7 build on M5's
    inputs (Test 3 passes on the series being forecast) or on M4's. VIX and the event calendar come
    from the panel. freq: monthly forecasts of the next 22 days, or weekly ones of the next 5 (WEEK)."""
    if freq == WEEK:
        x = lambda p: weekly_exog(panel, p)
        fs = {
            "M0": weekly.rolling,
            "M1": weekly.ewma,
            "M2": lambda p: garch_weekly(p, False, garch_fits(p, False)[1]),
            "M3": lambda p: garch_weekly(p, True, garch_fits(p, True)[1]),
            "M4": lambda p: har_weekly(p, False)[0],
            "M5": lambda p: har_weekly(p, True)[0],
            "M6": lambda p: har_weekly(p, down, x(p))[0],
            "M7": lambda p: boost_weekly(p, down, x(p))[0],
        }
        for name, f in fs.items():
            if getattr(f, "__name__", "") == "<lambda>":
                f.__name__ = f"{name}_weekly"
        return fs
    x = lambda p: exog(panel, p)
    fs = {
        "M0": filters.rolling,
        "M1": filters.ewma,
        "M2": lambda p: garch_fits(p, False)[0],
        "M3": lambda p: garch_fits(p, True)[0],
        "M4": lambda p: har_fits(p, False)[0],
        "M5": lambda p: har_fits(p, True)[0],
        "M6": lambda p: har_fits(p, down, x(p))[0],
        "M7": lambda p: boost_fits(p, down, x(p))[0],
    }
    for name, f in fs.items():
        if f.__name__ == "<lambda>":
            f.__name__ = name
    return fs


def oracle(p: pd.Series) -> pd.Series:
    """The holding month's actual variance, known in advance (the engine's oracle), from the same
    first month as the models: a look-ahead upper bound for Scorecard B, not a candidate."""
    o, _ = schedule(p.dropna())
    return engine_forecast.oracle(p).reindex(o.index)


oracle.looks_ahead = True
