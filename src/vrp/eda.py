"""
Track C, Phase C0: the EDA and its gate
=======================================
Everything reads development data only and, at each month-end t, only data up to that close.

    implied(panel)                    (VIX/100)² / 12 at each month's last trading day: the market's variance for
                                      the next month, in the forecasts' units (a month's Σ r²)
    vrp(panel, forecasts)             the implied variance, each model's forecast of the next 22 days, and their gap
    ahead(mon, h)                     the excess return over months t+1 .. t+h, compounded: Π(1 + r + rf) - Π(1 + rf)
    predictive(x, y, h)               y = a + b x by OLS, Newey-West errors with 6 + h - 1 lags, one-sided p for the
                                      expected sign; the slope in each of three equal eras; in- and out-of-sample R²
    oos_r2(x, y, h)                   the expanding-window out-of-sample R² (Campbell and Thompson): at each t, the
                                      regression and the mean fitted on the pairs complete by t (s + h <= t), from
                                      60 pairs on
    tests(vrps, y_by_h)               every model and horizon, with Holm across them
    gate(T)                           the cells passing all three of the gate's conditions, and whether any does
    describe(V), episodes(V)          question 1: the VRP's level, sign, persistence, and its months around 1998,
                                      2001 and 2008
    accuracy(forecasts, p)            question 4: each model's mean QLIKE against the next 22 days' realized variance
    monthly_put(put, panel)           question 5: the PutWrite index's monthly excess returns, from its first month
                                      with a value on every trading day

Rules in TRACK_C (PREREGISTRATION Entry 18).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import variance as v
from src.config import DEV_END
from src.engine.data import MONTH
from src.models.inputs import qlike, target
from src.walkforward import stats
from src.vrp import TRACK_C


def month_ends(index: pd.DatetimeIndex) -> pd.Series:
    """The last trading day of each month of a daily index, by month."""
    return pd.Series(index, index=index).groupby(index.to_period(MONTH)).max()


def implied(panel: pd.DataFrame) -> pd.Series:
    vix = panel["vix"].reindex(month_ends(panel.index).to_numpy())
    out = pd.Series(((vix / 100) ** 2 / 12).to_numpy(), index=vix.index.to_period(MONTH), name="implied")
    return out.dropna()


def vrp(panel: pd.DataFrame, forecasts: dict[str, pd.Series], first: str = TRACK_C["first_month"]) -> pd.DataFrame:
    """One row per month from `first` to the end of the panel: implied, forecast_<m> and vrp_<m> for each model.
    A forecast in row t is made at t's close for the next 22 trading days, as the implied variance is."""
    iv = implied(panel)
    out = pd.DataFrame({"implied": iv})
    for m, f in forecasts.items():
        out[f"forecast_{m}"] = f.reindex(out.index)
        out[f"vrp_{m}"] = out["implied"] - out[f"forecast_{m}"]
    last = pd.Period(panel.index[-1], MONTH)
    out = out.loc[pd.Period(first, MONTH):last]
    if out.isna().any().any():
        raise ValueError(f"a month from {first} lacks the implied variance or a forecast: {out.index[out.isna().any(axis=1)][0]}")
    return out


def ahead(mon: pd.DataFrame, h: int) -> pd.Series:
    """mon: monthly ret (excess) and rf. Row t: the excess return over t+1 .. t+h; NaN where those months are missing."""
    full = mon.reindex(pd.period_range(mon.index[0], mon.index[-1], freq=MONTH))
    grow = lambda x: sum(np.log1p(x.shift(-k)) for k in range(1, h + 1))
    out = np.exp(grow(full["ret"] + full["rf"])) - np.exp(grow(full["rf"]))
    return out.rename(f"ahead_{h}")


def lags(h: int) -> int:
    return 6 + h - 1


def oos_predictions(x: pd.Series, y: pd.Series, h: int, min_pairs: int = TRACK_C["oos_min_pairs"]) -> pd.DataFrame:
    """At each month t with min_pairs complete pairs before it: the regression's prediction of y_t from x_t, the
    training mean, and y_t. A pair s is complete at t's close when its h months have ended: s + h <= t."""
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    xs, ys, idx = d["x"].to_numpy(), d["y"].to_numpy(), d.index
    rows = {}
    for i, t in enumerate(idx):
        train = (idx + h) <= t
        if train.sum() < min_pairs:
            continue
        b, a = np.polyfit(xs[train], ys[train], 1)
        rows[t] = {"pred": a + b * xs[i], "bench": ys[train].mean(), "actual": ys[i]}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["pred", "bench", "actual"])


def oos_r2(x: pd.Series, y: pd.Series, h: int, min_pairs: int = TRACK_C["oos_min_pairs"]) -> dict:
    P = oos_predictions(x, y, h, min_pairs)
    if P.empty:
        return {"oos_months": 0, "oos_first": None, "r2_oos": np.nan}
    r2 = 1 - ((P["actual"] - P["pred"]) ** 2).sum() / ((P["actual"] - P["bench"]) ** 2).sum()
    return {"oos_months": len(P), "oos_first": str(P.index[0]), "r2_oos": float(r2)}


def predictive(x: pd.Series, y: pd.Series, h: int, sign: int = TRACK_C["sign"], eras: int = TRACK_C["eras"]) -> dict:
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    res = v.fit(d["y"], d["x"], lags(h))
    t = res.tvalues["x"]
    out = {"months": int(res.nobs), "first": str(d.index[0]), "last": str(d.index[-1]), "lags": lags(h),
           "slope": res.params["x"], "t": t, "p": float(norm.sf(sign * t)), "r2": res.rsquared}
    for k, part in enumerate(np.array_split(np.arange(len(d)), eras), start=1):
        e = d.iloc[part]
        out[f"era{k}"] = f"{e.index[0]} to {e.index[-1]}"
        out[f"era{k}_slope"] = np.polyfit(e["x"], e["y"], 1)[0]
    out["eras_hold"] = bool(all(sign * out[f"era{k}_slope"] > 0 for k in range(1, eras + 1)))
    return out | oos_r2(d["x"], d["y"], h)


def tests(vrps: pd.DataFrame, y_by_h: dict[int, pd.Series], models=TRACK_C["models"]) -> pd.DataFrame:
    """Every model and horizon, then Holm across all of them (the gate's 9 tests on the market)."""
    rows = [{"model": m, "h": h, **predictive(vrps[f"vrp_{m}"], y, h)} for m in models for h, y in y_by_h.items()]
    T = pd.DataFrame(rows)
    T["p_holm"] = stats.holm(T["p"].to_numpy())
    return T


def gate(T: pd.DataFrame, sign: int = TRACK_C["sign"], level: float = TRACK_C["level"]) -> tuple[pd.DataFrame, bool]:
    """Each cell's three conditions, and whether at least one cell meets all three."""
    G = T[["model", "h", "slope", "p_holm", "eras_hold", "r2_oos"]].copy()
    G["sign_holm"] = (sign * G["slope"] > 0) & (G["p_holm"] < level)
    G["oos_positive"] = G["r2_oos"] > 0
    G["passes"] = G["sign_holm"] & G["eras_hold"] & G["oos_positive"]
    return G, bool(G["passes"].any())


def describe(V: pd.DataFrame, models=TRACK_C["models"], units: float = TRACK_C["units"]) -> pd.DataFrame:
    """Question 1, per model, in %² a month: mean, median, standard deviation, the share of months positive and
    the AR(1) slope of the monthly VRP; the implied variance and the forecast's means beside it."""
    rows = []
    for m in models:
        x = V[f"vrp_{m}"]
        rows.append({"model": m, "months": len(x), "mean": x.mean() * units, "median": x.median() * units,
                     "std": x.std() * units, "share_positive": (x > 0).mean(),
                     "ar1": v.fit(x.shift(-1), x).params.iloc[1],
                     "implied_mean": V["implied"].mean() * units, "forecast_mean": V[f"forecast_{m}"].mean() * units})
    return pd.DataFrame(rows)


def episodes(V: pd.DataFrame, units: float = TRACK_C["units"], spans=TRACK_C["episodes"]) -> pd.DataFrame:
    parts = [V.loc[pd.Period(a, MONTH):pd.Period(b, MONTH)].mul(units).assign(episode=name) for name, (a, b) in spans.items()]
    return pd.concat(parts).rename_axis("month").reset_index()


def accuracy(forecasts: dict[str, pd.Series], p: pd.Series, months: pd.PeriodIndex) -> pd.DataFrame:
    """Question 4: mean QLIKE of each model against the realized variance of the next 22 trading days, on the
    months every model and the target cover."""
    y = target(p)
    F = pd.DataFrame(forecasts).reindex(months)
    ok = F.notna().all(axis=1) & y.reindex(months).notna()
    L = pd.DataFrame(qlike(y.reindex(months)[ok].to_numpy()[:, None], F[ok].to_numpy()), columns=F.columns)
    return pd.DataFrame({"model": F.columns, "qlike": L.mean().to_numpy(), "months": int(ok.sum()),
                         "accuracy_rank": L.mean().rank().astype(int).to_numpy()})


def monthly_put(put: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The PutWrite index's monthly excess return (its month-end level over the last, minus the month's compounded
    rf), from the first month whose every trading day of the panel has a value. ret and rf, as mon tables."""
    days = panel.index[(panel.index >= put.index[0].to_period(MONTH).start_time) & (panel.index <= DEV_END)]
    have = pd.Series(days.isin(put.index), index=days).groupby(days.to_period(MONTH)).all()
    first = have.index[have.to_numpy()][0]
    if not have.loc[first:].all():
        raise ValueError(f"the PutWrite index misses a trading day in {have.loc[first:].index[~have.loc[first:].to_numpy()][0]}")
    ends = month_ends(days).loc[first:]
    level = put["put"].reindex(ends.to_numpy())
    level.index = ends.index
    cash = np.exp(np.log1p(panel["rf"]).groupby(panel.index.to_period(MONTH)).sum()) - 1
    out = pd.DataFrame({"ret": level.pct_change() - cash.reindex(level.index), "rf": cash.reindex(level.index)})
    return out.iloc[1:]
