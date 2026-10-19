"""
Phase 4: portfolio construction
===============================
Combine the survivors into one portfolio, then time that single portfolio.

    recipes(panel, track)          -> one recipe per year-end and mix version, from data up to that date
    returns(panel, track, recipes) -> daily combined return p, serving size and managed return
    tests(p)                       -> Phase 2 Tests 1 and 4 on a combined return

A recipe is set at each year-end and held for the whole next calendar year:

  1. hedge ratios: an ETF standing in for a factor tilt is traded as the tilt,
     ETF minus beta x SPY, with beta from the estimation window;
  2. duplicates: of two assets whose monthly returns correlate above +0.7 (on all months
     or on crisis months), the one with the shorter history is dropped;
  3. mix weights w_i, by one of four methods (METHODS);
  4. c, which matches the managed and unmanaged variance over the estimation window.

Within the year only the serving size changes: each month, scale = c / RV of p last month,
capped at 1.5. Exposure to asset i is scale x w_i. Mix weights only set the ratio between the bets; the
total size comes from the scale.

Development sample only: everything here takes the output of src.data.load.load_dev().
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from src import variance as v
from src.frozen import freeze
from src.screen import SCREEN_RULES, year_ends

# ---------------------------------------------------------------- rules
# Fixed before any Phase 4 result was looked at, except two dated changes to Step 1 (see dedupe),
# both made on 2026-10-17 after the first run and neither changing a result. Tests 1 and 4 use
# the Phase 2 thresholds in variance.RULES; Test 1 gates on level phi, log phi is reported beside it.
PORTFOLIO_RULES = freeze({
    "max_corr": 0.7,          # Step 1: a pair whose monthly ρ is above +0.7 is a duplicate ...
    "crisis_share": 0.10,     #   ... on all months, or on the 10% of months with the highest market RV,
                              #   judged only on min_months common months
    "window_months": 120,     # Step 2: weights, hedge ratios and c from the last 10 years (all of it if shorter)
    "min_months": SCREEN_RULES["min_months"],   # paper tracks: 10 years before the first recipe; and
                                                # Tests 1 and 4 on p are judged only with 120 months of p
    "etf_min_months": 12,     # ETF track: one year before the first recipe (Phase 3's etf_min_days = 252)
    "max_size": 1.5,          # Step 3: serving size capped at 1.5x in the primary test (set in the plan);
                              #   Phase 8 re-runs with 1.0, 2.0 and no cap
})
# The pass rule: Tests 1 and 4 both pass on p of the primary track and mix. Every other
# track and mix is reported, not judged. If the primary fails, it stays the primary.
# The primary includes paper BAB, which cannot be traded, so its results are an upper bound.
PRIMARY = ("paper", "inv_vol")

# Survivors of the Phase 3 screen on the full development sample. "paper" holds the three factors
# that pass every factor filter; betting-against-beta has no tradeable ETF, so "etf" is the
# tradeable version (SPY and MTUM's market-hedged tilt) and "paper_no_bab" its long-history twin.
# A hedged asset is traded as its tilt: hedge {"mtum": "spy"} means mtum - beta x spy.
TRACKS = freeze({
    "paper": {"assets": ["mkt", "mom", "bab"], "hedge": {}, "min_months": PORTFOLIO_RULES["min_months"]},
    "paper_no_bab": {"assets": ["mkt", "mom"], "hedge": {}, "min_months": PORTFOLIO_RULES["min_months"]},
    "etf": {"assets": ["spy", "mtum"], "hedge": {"mtum": "spy"}, "min_months": PORTFOLIO_RULES["etf_min_months"]},
})
METHODS = ["equal", "inv_var", "inv_vol", "shrunk_mv"]


# ---------------------------------------------------------------- Step 1: duplicates

def hedge_betas(x: pd.DataFrame, hedge: dict[str, str]) -> dict[str, float]:
    """OLS slope of each hedged asset on its hedge instrument."""
    return {a: x[a].cov(x[h]) / x[h].var() for a, h in hedge.items()}


def hedged(x: pd.DataFrame, hedge: dict[str, str], betas: dict[str, float]) -> pd.DataFrame:
    """Replace each hedged asset by its tilt: asset - beta x hedge instrument."""
    x = x.copy()
    for a, h in hedge.items():
        x[a] = x[a] - betas[a] * x[h]
    return x


def realtime_tilt(panel: pd.DataFrame, etf: str, hedge: str = "spy") -> pd.Series:
    """etf - beta x hedge, beta set at each year-end from the estimation window and held for the
    next year (same window and minimum history as the ETF track)."""
    r = panel[[etf, hedge]].dropna()
    out = []
    for asof in year_ends(r.index):
        hist = r.loc[:asof]
        hold = r.loc[r.index.year == asof.year + 1]
        if hist.index.to_period("M").nunique() < PORTFOLIO_RULES["etf_min_months"] or hold.empty:
            continue
        win = hist.loc[hist.index > asof - pd.DateOffset(months=PORTFOLIO_RULES["window_months"])]
        out.append(hedged(hold, {etf: hedge}, hedge_betas(win, {etf: hedge}))[etf])
    return pd.concat(out).rename(f"{etf} tilt")


def monthly_returns(x: pd.DataFrame) -> pd.DataFrame:
    """Compounded calendar-month returns, one column per asset (Phase 2's monthly table)."""
    return pd.DataFrame({a: v.monthly_table(x[a])["ret"] for a in x})


def correlations(mon: pd.DataFrame, mkt_rv: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Correlation of monthly returns over all common months, and over the crisis months: the
    10% of those months with the highest market RV. Returns (all, crisis, number of crisis months)."""
    mon = mon.dropna()
    rv = mkt_rv.reindex(mon.index)
    crisis = rv >= rv.quantile(1 - PORTFOLIO_RULES["crisis_share"])
    return mon.corr(), mon[crisis].corr(), int(crisis.sum())


def dedupe(mon: pd.DataFrame, mkt_rv: pd.Series, history: dict[str, int], strength) -> dict[str, str]:
    """Assets to drop, with the reason. Pairs above the cutoff on all months or crisis months are
    taken most correlated first; of each, the one with the shorter history is dropped, and if the
    histories are equal, the one with the weaker Phase 2 Test 4 result (strength: asset -> t-stat).

    Two changes of 2026-10-17, after the first run:
    - pairs are judged only on min_months common months, so the crisis set has at least 12 months.
      With fewer, nothing is dropped (the first run judged a 2-month crisis set, whose |ρ| is always 1);
    - only positive correlations count. The first version cut on |ρ|, so a strongly negative pair,
      which is a hedge rather than the same bet, would have been cut (market-momentum reached -0.69)."""
    if len(mon.dropna()) < PORTFOLIO_RULES["min_months"]:
        return {}
    full, crisis, _ = correlations(mon, mkt_rv)
    worst = np.maximum(full, crisis)
    pairs = sorted(((worst.loc[a, b], a, b) for a, b in combinations(mon.columns, 2)), reverse=True)
    dropped = {}
    for rho, a, b in pairs:
        if rho <= PORTFOLIO_RULES["max_corr"] or a in dropped or b in dropped:
            continue
        keep, drop = sorted((a, b), key=lambda k: history[k], reverse=True)
        if history[a] == history[b]:
            keep, drop = sorted((a, b), key=strength, reverse=True)
        dropped[drop] = f"ρ {rho:+.2f} with {keep}"
    return dropped


# ---------------------------------------------------------------- Step 2: mix weights

def mix_weights(x: pd.DataFrame, method: str) -> pd.Series:
    """Mix weights from daily returns in the estimation window. No negatives; they sum to one.

    equal      1/N
    inv_var    proportional to 1 / sigma²
    inv_vol    proportional to 1 / sigma (the primary)
    shrunk_mv  average of 1/N and the normalized Σ⁻¹μ, with Ledoit-Wolf Σ and negatives set to zero
    """
    n = x.shape[1]
    if method == "equal":
        w = pd.Series(1.0, x.columns)
    elif method == "inv_var":
        w = 1 / x.var()
    elif method == "inv_vol":
        w = 1 / x.std()
    elif method == "shrunk_mv":
        cov = LedoitWolf().fit(x.to_numpy()).covariance_
        mv = pd.Series(np.linalg.solve(cov, x.mean().to_numpy()), x.columns).clip(lower=0)
        mv = mv / mv.sum() if mv.sum() > 0 else pd.Series(1 / n, x.columns)   # every mean <= 0: 1/N
        w = (mv + 1 / n) / 2
    else:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    return w / w.sum()


def vol_match(mon: pd.DataFrame) -> float:
    """c such that (c / RV_t) r_{t+1} has the variance of r_{t+1}: the c of variance.t4_spanning."""
    df = pd.DataFrame({"f": mon["ret"].shift(-1), "rv": mon["rv"]}).dropna()
    return df["f"].std() / (df["f"] / df["rv"]).std()


# ---------------------------------------------------------------- recipes and returns

def recipes(panel: pd.DataFrame, track: str | dict) -> pd.DataFrame:
    """One recipe per year-end and mix version, each from data up to that year-end only.

    track: a key of TRACKS, or a spec of the same form (Phase 8 drops one asset at a time).
    Columns: w_<asset> mix weights, beta_<asset> hedge ratios, pos_<instrument> positions in the
    traded instruments per unit of serving size, c, the assets dropped as duplicates, and the
    correlations behind that decision (rho_<a>_<b> on all months, crisis_<a>_<b> on crisis months).
    """
    spec = TRACKS[track] if isinstance(track, str) else track
    cols, hedge = spec["assets"], spec["hedge"]
    r = panel[cols].dropna()
    mkt_rv = v.monthly_table(panel["mkt"])["rv"]
    rows = []
    for asof in year_ends(r.index):
        hist = r.loc[:asof]
        if hist.index.to_period("M").nunique() < spec["min_months"]:
            continue
        win = hist.loc[hist.index > asof - pd.DateOffset(months=PORTFOLIO_RULES["window_months"])]
        betas = hedge_betas(win, hedge)
        x_hist, x_win = hedged(hist, hedge, betas), hedged(win, hedge, betas)

        mon = monthly_returns(x_hist)                            # duplicates: all history so far
        history = {a: int(panel[a].loc[:asof].count()) for a in cols}
        dropped = dedupe(mon, mkt_rv, history, lambda a: v.t4_spanning(v.monthly_table(x_hist[a]))["t_alpha"])
        full, crisis, _ = correlations(mon, mkt_rv)
        rho = {"corr_months": len(mon.dropna())}
        for a, b in combinations(cols, 2):
            rho[f"rho_{a}_{b}"], rho[f"crisis_{a}_{b}"] = full.loc[a, b], crisis.loc[a, b]
        keep = [a for a in cols if a not in dropped]

        for method in METHODS:
            w = mix_weights(x_win[keep], method).reindex(cols, fill_value=0.0)
            pos = w.copy()
            for a, h in hedge.items():                           # the tilt is a short position in h
                pos[h] -= w[a] * betas[a]
            rows.append({"asof": asof, "method": method, "window_start": win.index[0],
                         **w.add_prefix("w_"), **{f"beta_{a}": b for a, b in betas.items()},
                         **pos.add_prefix("pos_"), "c": vol_match(v.monthly_table(x_win @ w)),
                         "dropped": "; ".join(f"{a} ({why})" for a, why in dropped.items()), **rho})
    return pd.DataFrame(rows).set_index(["asof", "method"])


def returns(panel: pd.DataFrame, track: str, rec: pd.DataFrame,
            max_size: float | None = PORTFOLIO_RULES["max_size"]) -> pd.DataFrame:
    """Daily combined return p, serving size and managed return for each mix version.

    Each day uses the recipe set at the last year-end before it. The serving size for a month is
    c / RV of p in the month before, with p under the recipe in force (so January's size uses
    December's returns under the new recipe), capped at max_size (None: no cap). c is the
    uncapped variance match, so with a cap the timed mix runs below p's volatility.
    Columns: (method, "p" | "scale" | "managed").
    """
    spec = TRACKS[track]
    cols, hedge = spec["assets"], spec["hedge"]
    r = panel[cols].dropna()
    out = {m: [] for m in METHODS}
    for asof, grp in rec.groupby(level="asof"):
        hold = r.index[r.index.year == asof.year + 1]
        if hold.empty:
            continue
        span = r.loc[asof.replace(day=1):hold[-1]]             # starts with the recipe month, for its RV
        for method, row in grp.droplevel("asof").iterrows():
            betas = {a: row[f"beta_{a}"] for a in hedge}
            p = hedged(span, hedge, betas) @ row[[f"w_{a}" for a in cols]].astype(float).set_axis(cols)
            scale = row["c"] / v.monthly_table(p)["rv"]
            scale.index = scale.index + 1                       # set at month-end, used the month after
            if max_size is not None:
                scale = scale.clip(upper=max_size)
            s = scale.reindex(hold.to_period("M")).to_numpy()
            out[method].append(pd.DataFrame({"p": p.loc[hold], "scale": s, "managed": s * p.loc[hold]}, index=hold))
    return pd.concat({m: pd.concat(f) for m, f in out.items()}, axis=1)


# ---------------------------------------------------------------- Step 3: tests on p

def tests(p: pd.Series) -> dict:
    """Phase 2 Tests 1 and 4 on a combined daily return. Judged only with min_months of p.
    phi_log (AR(1) slope of log RV) is information only: the rule is on phi."""
    p = p.dropna()
    mon = v.monthly_table(p)
    t1 = v.t1_persistence(p, mon)
    span = v.t4_spanning(mon)
    t4 = v.t4_verdict(v.t4_sort(mon), span)
    judged = len(mon) >= PORTFOLIO_RULES["min_months"]
    return {"start": p.index[0].date(), "months": len(mon), "p20": t1["p20"], "p60": t1["p60"], "phi": t1["phi"],
            "phi_log": t1["phi_log"], "t1": t1["pass"], "spearman": t4["spearman"], "sort_ok": t4["sort_ok"],
            "alpha": span["alpha"], "t_alpha": span["t_alpha"], "t4": t4["pass"], "judged": judged,
            "pass": bool(t1["pass"] and t4["pass"]) if judged else None}
