"""
Phase 2: how each asset's variance behaves
==========================================
Variance series (Step 1) and the ten diagnostic tests. Development sample only:
everything here takes the output of src.data.load.load_dev().

Every test function takes one asset's daily returns (and/or its monthly table)
and returns plain numbers, so the report notebook can loop over assets and
tabulate. Pass/fail thresholds live in RULES and were fixed before any test
was run (Phase 2 rule: write each decision rule before looking).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import acf, pacf

from src.config import EVENTS, FACTORS

# ---------------------------------------------------------------- decision rules
# Fixed before running any test. |t| > 1.96 is the 5% two-sided threshold.
RULES = {
    "t1_ljung_box_p": 0.01,      # T1: Ljung-Box on r² rejects at 1%, at 20 and 60 lags ...
    "t1_phi": 0.5,               #     ... and monthly AR(1) slope on RV above 0.5 (the gate)
    "t": 1.96,                   # T2/T3/T4/T5/T9: significance threshold on |t|
    "t4_spearman": -0.7,         # T4: rank correlation of r̄/s² with quintile at most -0.7
    "t5_event_ratio": 1.2,       # T5: event-day mean r² at least 1.2x other days
    "t5_vix_dr2": 0.01,          # T5: VIX adds at least 1 point of R²
    "t6_min_share": 1 / 3,       # T6: every era same sign and >= 1/3 of full-sample size
    "t7_log_skew": 0.5,          # T7: log RV near normal if |skew| < 0.5 ...
    "t7_log_kurt": 4.0,          #     ... and kurtosis < 4 (normal = 3)
    "t8_vif": 10.0,              # T8: VIF above 10 is severe
    "t10_dr2": 0.005,            # T10A: history stops helping when OOS R² gains < 0.5 point
    "t10_drift_share": 0.20,     # T10B: phi drifts if rolling phi leaves its 95% band > 20% of the time
}

NW_DAILY = 44            # Newey-West lags for 22-day overlapping targets (spec: at least 22)
NW_MONTHLY = 6           # Newey-West lags for non-overlapping monthly regressions
SUBPERIODS = {"1926-1962": ("1926", "1962"), "1963-1990": ("1963", "1990"),
              "1991-2018": ("1991", "2018")}
ASSETS = FACTORS + ["ew7"]
STAGES = {"1 day": ["d"], "1-5 days": ["d", "w"], "1-22 days": ["d", "w", "m"],
          "1-66 days": ["d", "w", "m", "q"]}


# ---------------------------------------------------------------- Step 1: series

def daily_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Factor excess returns plus ew7: all seven factors equally weighted, rebalanced
    daily, from the first day all seven exist (1963-07). ew7 is illustrative only:
    portfolio weights are chosen in a later phase."""
    r = panel[FACTORS].copy()
    r["ew7"] = panel[FACTORS].mean(axis=1, skipna=False)
    return r


def monthly_table(r: pd.Series) -> pd.DataFrame:
    """Calendar-month table from daily returns: compounded return, realized variance
    RV = sum r², downside RV (Wang-Yan: window extended back one month when the
    month has fewer than three down days), upside RV, log RV."""
    r = r.dropna()
    d = pd.DataFrame({"r2": r ** 2, "r2_dn": r.clip(upper=0) ** 2, "dn": r < 0, "lr": np.log1p(r)})
    m = d.groupby(r.index.to_period("M")).agg(
        rv=("r2", "sum"), dn_sum=("r2_dn", "sum"), n_dn=("dn", "sum"), lr=("lr", "sum"), n=("r2", "size"))
    m = m[m["n"] >= 10]                                   # drops a partial first month, if any
    return pd.DataFrame({
        "ret": np.expm1(m["lr"]),
        "rv": m["rv"],
        "rv_down": m["dn_sum"].where(m["n_dn"] >= 3, m["dn_sum"] + m["dn_sum"].shift(1)),
        "rv_up": m["rv"] - m["dn_sum"],
        "log_rv": np.log(m["rv"]),
        "n": m["n"],
        "extended": m["n_dn"] < 3,
    })


def parkinson(ohlc: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily variance from the high-low range (Parkinson 1980)."""
    o = ohlc[ohlc["ticker"] == ticker].set_index("date")
    return ((np.log(o["high"]) - np.log(o["low"])) ** 2 / (4 * np.log(2))).rename(ticker)


def har_frame(v: pd.Series, target: pd.Series | None = None) -> pd.DataFrame:
    """Daily HAR design. y = mean of target over the next 22 days (t+1..t+22);
    d, w, m, q = v today and its mean over the last 5, 22, 66 days.
    v is the daily variance proxy (r² or Parkinson); target defaults to v."""
    target = v if target is None else target
    return pd.DataFrame({"y": target.rolling(22).mean().shift(-22), "d": v,
                         "w": v.rolling(5).mean(), "m": v.rolling(22).mean(),
                         "q": v.rolling(66).mean()})


def fit(y: pd.Series, X: pd.DataFrame | pd.Series, cov: str | int | None = None):
    """OLS of y on X plus a constant. cov: None (plain), 'white', or Newey-West lags."""
    X = X.to_frame() if isinstance(X, pd.Series) else X
    df = pd.concat([y.rename("_y"), X], axis=1).dropna()
    model = sm.OLS(df["_y"], sm.add_constant(df.drop(columns="_y")))
    if cov is None:
        return model.fit()
    if cov == "white":
        return model.fit(cov_type="HC0")
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": cov})


# ---------------------------------------------------------------- Test 1

def t1_persistence(r: pd.Series, mon: pd.DataFrame) -> dict:
    lb = acorr_ljungbox(r.dropna() ** 2, lags=[20, 60])
    ar = fit(mon["rv"].shift(-1), mon["rv"])
    ar_log = fit(mon["log_rv"].shift(-1), mon["log_rv"])
    phi = ar.params["rv"]
    return {"months": int(ar.nobs), "Q20": lb["lb_stat"].iloc[0], "p20": lb["lb_pvalue"].iloc[0],
            "Q60": lb["lb_stat"].iloc[1], "p60": lb["lb_pvalue"].iloc[1],
            "phi": phi, "phi_se": ar.bse["rv"], "phi_log": ar_log.params["log_rv"], "R2": ar.rsquared,
            "pass": bool(lb["lb_pvalue"].max() < RULES["t1_ljung_box_p"] and phi > RULES["t1_phi"])}


# ---------------------------------------------------------------- Test 2

def t2_har(r: pd.Series) -> dict:
    h = har_frame(r.dropna() ** 2)[["y", "d", "w", "m"]].dropna()
    out = {}
    for name, cols in (("d", ["d"]), ("d+w", ["d", "w"]), ("d+w+m", ["d", "w", "m"])):
        res = fit(h["y"], h[cols], NW_DAILY)
        out[f"R2 {name}"] = res.rsquared
    for k in ("d", "w", "m"):
        out[f"b_{k}"], out[f"t_{k}"] = res.params[k], res.tvalues[k]
    out["long_memory"] = bool(all(res.params[k] > 0 and res.tvalues[k] > RULES["t"] for k in ("w", "m")))
    return out


def autocorrelations(r: pd.Series, nlags: int = 100) -> pd.DataFrame:
    r = r.dropna()
    return pd.DataFrame({"acf r²": acf(r ** 2, nlags=nlags, fft=True),
                         "acf |r|": acf(r.abs(), nlags=nlags, fft=True),
                         "pacf r²": pacf(r ** 2, nlags=nlags, method="ywm")}).iloc[1:]


# ---------------------------------------------------------------- Test 3

def t3_asymmetry(r: pd.Series) -> dict:
    r = r.dropna()
    X = pd.DataFrame({"r2": r ** 2, "r2_neg": r ** 2 * (r < 0)})
    res = fit((r ** 2).rolling(22).mean().shift(-22), X, NW_DAILY)
    delta, t = res.params["r2_neg"], res.tvalues["r2_neg"]
    return {"k1": res.params["r2"], "t_k1": res.tvalues["r2"], "delta": delta, "t_delta": t,
            "pass": bool(delta > 0 and t > RULES["t"])}


def t3_return_side(mon: pd.DataFrame) -> dict:
    """Next month's return on this month's downside and upside RV (White errors)."""
    res = fit(mon["ret"].shift(-1), mon[["rv_down", "rv_up"]], "white")
    return {"b_down": res.params["rv_down"], "t_down": res.tvalues["rv_down"],
            "b_up": res.params["rv_up"], "t_up": res.tvalues["rv_up"]}


# ---------------------------------------------------------------- Test 4

def t4_sort(mon: pd.DataFrame) -> pd.DataFrame:
    """Months in quintiles of RV_t; next month's annualised mean, variance, r̄/s², Sharpe."""
    df = pd.DataFrame({"x": mon["rv"], "y": mon["ret"].shift(-1)}).dropna()
    df["q"] = pd.qcut(df["x"], 5, labels=[1, 2, 3, 4, 5])
    g = df.groupby("q", observed=True)["y"]
    tab = pd.DataFrame({"mean": g.mean() * 12, "var": g.var() * 12})
    tab["ratio"] = tab["mean"] / tab["var"]
    tab["sharpe"] = tab["mean"] / np.sqrt(tab["var"])
    return tab


def t4_spanning(mon: pd.DataFrame) -> dict:
    """(c / RV_t) r_{t+1} = alpha + beta r_{t+1}; c matches the managed and raw volatility."""
    df = pd.DataFrame({"f": mon["ret"].shift(-1), "rv": mon["rv"]}).dropna()
    raw = df["f"] / df["rv"]
    c = df["f"].std() / raw.std()
    res = fit(c * raw, df["f"], "white")
    return {"alpha": res.params["const"] * 12, "t_alpha": res.tvalues["const"], "beta": res.params["f"]}


def t4_verdict(tab: pd.DataFrame, span: dict) -> dict:
    rho = pd.Series(tab["ratio"].to_numpy()).corr(pd.Series(range(5)), method="spearman")
    sort_ok = tab["ratio"].iloc[-1] < tab["ratio"].iloc[0] and rho <= RULES["t4_spearman"]
    return {"spearman": rho, "sort_ok": bool(sort_ok),
            "pass": bool(sort_ok and span["alpha"] > 0 and span["t_alpha"] > RULES["t"])}


# ---------------------------------------------------------------- Test 5

def t5_events(r: pd.Series, flags: pd.DataFrame) -> pd.DataFrame:
    """Mean r² on each event's days vs days with no event (1994+), Newey-West t."""
    df = pd.concat([r.rename("r"), flags[EVENTS]], axis=1).dropna().astype(float)
    df["r2"] = df["r"] ** 2 * 1e4                       # squared return in basis points², for readability
    res = fit(df["r2"], df[EVENTS], 22)
    base = df.loc[df[EVENTS].sum(axis=1) == 0, "r2"].mean()
    rows = {}
    for e in EVENTS:
        ratio = df.loc[df[e] == 1, "r2"].mean() / base
        rows[e] = {"days": int(df[e].sum()), "ratio": ratio, "t": res.tvalues[e],
                   "pass": bool(ratio >= RULES["t5_event_ratio"] and res.tvalues[e] > RULES["t"])}
    return pd.DataFrame(rows).T


def t5_vix(r: pd.Series, vix: pd.Series) -> dict:
    """Add VIX-implied daily variance, (VIX/100)²/252, to the d+w+m HAR (1990+)."""
    h = har_frame(r.dropna() ** 2)
    h["vix"] = ((vix / 100) ** 2 / 252).reindex(h.index)
    h = h[["y", "d", "w", "m", "vix"]].dropna()
    base, full = fit(h["y"], h[["d", "w", "m"]], NW_DAILY), fit(h["y"], h[["d", "w", "m", "vix"]], NW_DAILY)
    dr2 = full.rsquared - base.rsquared
    return {"R2 HAR": base.rsquared, "R2 HAR+VIX": full.rsquared, "dR2": dr2,
            "b_vix": full.params["vix"], "t_vix": full.tvalues["vix"],
            "pass": bool(full.tvalues["vix"] > RULES["t"] and dr2 >= RULES["t5_vix_dr2"])}


# ---------------------------------------------------------------- Test 6

def headline(r: pd.Series, mon: pd.DataFrame) -> dict:
    """One number per test (1-4), for comparing eras."""
    t2 = t2_har(r)
    return {"phi (T1)": t1_persistence(r, mon)["phi"], "b_w+b_m (T2)": t2["b_w"] + t2["b_m"],
            "delta (T3)": t3_asymmetry(r)["delta"], "alpha (T4)": t4_spanning(mon)["alpha"]}


def consistent(full: float, subs: list[float]) -> bool:
    return all(np.sign(s) == np.sign(full) and abs(s) >= RULES["t6_min_share"] * abs(full) for s in subs)


# ---------------------------------------------------------------- Test 7

def t7_moments(r: pd.Series, mon: pd.DataFrame) -> dict:
    out = {}
    for name, x in (("daily r", r.dropna()), ("monthly r", mon["ret"]), ("RV", mon["rv"]),
                    ("log RV", mon["log_rv"])):
        out[f"{name} skew"] = stats.skew(x)
        out[f"{name} kurt"] = stats.kurtosis(x, fisher=False)
    out["log RV near normal"] = bool(abs(out["log RV skew"]) < RULES["t7_log_skew"]
                                     and out["log RV kurt"] < RULES["t7_log_kurt"])
    return out


# ---------------------------------------------------------------- Test 8

def _vifs(X: pd.DataFrame) -> list[float]:
    A = sm.add_constant(X.dropna()).to_numpy()
    return [variance_inflation_factor(A, i + 1) for i in range(X.shape[1])]


def t8_vif(r: pd.Series) -> dict:
    v = r.dropna() ** 2
    h = har_frame(v)
    pieces = pd.DataFrame({"d": v, "w_ex": v.shift(1).rolling(4).mean(),     # the 4 days before today
                           "m_ex": v.shift(5).rolling(17).mean()})          # the 17 days before the week
    a, b = _vifs(h[["d", "w", "m"]]), _vifs(pieces)
    return {"VIF d": a[0], "VIF w": a[1], "VIF m": a[2],
            "VIF d (pieces)": b[0], "VIF w_ex": b[1], "VIF m_ex": b[2],
            "severe": bool(max(a) > RULES["t8_vif"])}


# ---------------------------------------------------------------- Test 9

def t9_curvature(mon: pd.DataFrame) -> dict:
    x = mon["log_rv"] - mon["log_rv"].mean()               # centred: same c and t, better conditioning
    df = pd.DataFrame({"y": mon["log_rv"].shift(-1), "x": x}).dropna()
    quad = fit(df["y"], pd.DataFrame({"x": df["x"], "x2": df["x"] ** 2}), NW_MONTHLY)
    top = (df["x"] >= df["x"].quantile(0.8)).astype(float)
    split = fit(df["y"], pd.DataFrame({"x": df["x"], "top": top, "x_top": df["x"] * top}), NW_MONTHLY)
    return {"c": quad.params["x2"], "t_c": quad.tvalues["x2"],
            "slope": split.params["x"], "slope top 20%": split.params["x"] + split.params["x_top"],
            "t_shift": split.tvalues["x_top"],
            "nonlinear": bool(abs(quad.tvalues["x2"]) > RULES["t"] or abs(split.tvalues["x_top"]) > RULES["t"])}


# ---------------------------------------------------------------- Test 10

def t10_oos_stages(r: pd.Series, train_years: int = 10) -> dict:
    """Out-of-sample R² of the 22-day variance forecast as longer history is added.
    Expanding window, refit each January on days whose 22-day target ended before it;
    benchmark is the expanding mean of the target."""
    h = har_frame(r.dropna() ** 2).dropna()
    y, years = h["y"].to_numpy(), h.index.year
    pred = {k: np.full(len(h), np.nan) for k in STAGES}
    bench = np.full(len(h), np.nan)
    for year in range(years[0] + train_years, years[-1] + 1):
        test = years == year
        train = np.arange(len(h)) <= np.argmax(test) - 23        # target t+1..t+22 known before Jan 1
        bench[test] = y[train].mean()
        for k, cols in STAGES.items():
            A = np.column_stack([np.ones(len(h)), h[cols].to_numpy()])
            beta = np.linalg.lstsq(A[train], y[train], rcond=None)[0]
            pred[k][test] = A[test] @ beta
    ok = ~np.isnan(bench)
    sse0 = ((y - bench)[ok] ** 2).sum()
    return {k: 1 - ((y - p)[ok] ** 2).sum() / sse0 for k, p in pred.items()}


def se_phi(phi: float, n: int) -> float:
    return np.sqrt((1 - phi ** 2) / n)


def rolling_phi(mon: pd.DataFrame, window: int = 120) -> pd.Series:
    x = mon["rv"].to_numpy()
    out = {}
    for end in range(window, len(x) + 1):
        z, y = x[end - window:end - 1], x[end - window + 1:end]
        out[mon.index[end - 1]] = np.cov(z, y)[0, 1] / z.var(ddof=1)
    return pd.Series(out)


def drift_share(mon: pd.DataFrame, phi: float, window: int = 120) -> float:
    band = 1.96 * se_phi(min(phi, 0.99), window)
    return float((rolling_phi(mon, window).sub(phi).abs() > band).mean())


def cusum_mean(x: pd.Series, lags: int = 12) -> dict:
    """CUSUM test for a shift in the mean, using a Newey-West long-run variance so that
    persistence alone is not read as a break. Under no break, max |S| follows the
    Kolmogorov distribution (5% critical value 1.358)."""
    x = x.dropna()
    e = (x - x.mean()).to_numpy()
    n = len(e)
    lrv = e @ e / n + 2 * sum((1 - j / (lags + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, lags + 1))
    s = np.abs(np.cumsum(e)) / np.sqrt(n * lrv)
    k = int(s.argmax())
    return {"stat": s[k], "p": stats.kstwobign.sf(s[k]), "most likely break": x.index[k]}


def ar1_oos(mon: pd.DataFrame, train: int = 120) -> float:
    """Out-of-sample R² of a monthly AR(1) on RV, expanding window, refit every month,
    against the expanding historical mean of RV."""
    x = mon["rv"].to_numpy()
    err, err0 = [], []
    for t in range(train, len(x) - 1):
        z, y = x[:t], x[1:t + 1]
        b = np.cov(z, y)[0, 1] / z.var(ddof=1)
        a = y.mean() - b * z.mean()
        err.append(x[t + 1] - (a + b * x[t]))
        err0.append(x[t + 1] - x[:t + 1].mean())
    return 1 - np.sum(np.square(err)) / np.sum(np.square(err0))


# ---------------------------------------------------------------- ETFs: Parkinson vs r²

def proxy_comparison(r: pd.Series, park: pd.Series) -> dict:
    """Forecast the next 22 days' mean r² with HAR terms built from r² or from Parkinson."""
    r = r.dropna()
    target = r ** 2
    out = {}
    for name, v in (("r²", target), ("Parkinson", park.reindex(r.index))):
        h = har_frame(v, target).dropna()
        out[f"R2 HAR ({name})"] = fit(h["y"], h[["d", "w", "m"]]).rsquared
    out["Parkinson / r² mean"] = park.reindex(r.index).mean() / target.mean()
    return out
