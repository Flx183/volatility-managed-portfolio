"""
Phase 3: asset screening
========================
Which assets give the strategy a real chance? Six filters, run at any as-of date
using only data up to that date:

    screen(panel, ohlc, asof) -> one row per candidate factor, with every filter's
                                 numbers, pass/fail flags and the reason for each cut

Phase 7's walk-forward calls it once a year (see year_ends) and lets an asset into
the portfolio only if `survivor` is True on that date.

Factors face filters 1, 2, 3 and 6 (their own return history). Filters 4 and 5 apply
to the ETF that stands in for the factor in trading (ETF_FOR). A factor with no ETF
in the data cannot be traded, so it is cut.

Filters 1 and 3 reuse the Phase 2 tests and thresholds (src/variance.py: RULES).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import variance as v
from src.config import FACTORS

# ---------------------------------------------------------------- rules
# Fixed before any Phase 3 result was looked at. Phase 2 thresholds (Ljung-Box 1%,
# phi > 0.5, t > 1.96, rank correlation <= -0.7) are in variance.RULES.
SCREEN_RULES = {
    "min_months": 120,        # a factor needs 10 years of complete months before it can be screened
    "n_eras": 3,              # F6: the history so far is cut into this many equal-length eras
    "etf_min_days": 252,      # F4, F5: an ETF needs a year of trading history to be judged
    "etf_r2": 0.7,            # F4: R² of the ETF on market, size, value, momentum above this
    "dollar_volume": 50e6,    # F5: median daily dollar volume above this ...
    "liquidity_days": 252,    #     ... over the last 252 trading days
}
# F5's second half, bid-ask spread below 5 bp, is NOT applied: Phase 1 has no quote data,
# and the high-low spread proxy (Corwin-Schultz) reads 8-15 bp for SPY, whose true spread is
# about 1 bp. Verify spreads with quote data in the cost phase.

# The ETF that stands in for each factor. A judgment, not a measurement: QUAL (quality) for
# profitability and USMV (minimum volatility) for betting-against-beta are loose matches.
# Size, value and investment have no ETF among the four downloaded.
ETF_FOR = {"mkt": "SPY", "mom": "MTUM", "rmw": "QUAL", "bab": "USMV"}
ETF_REGRESSORS = ["mkt", "smb", "hml", "mom"]


def year_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar year in the index (the last one may be a partial year)."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.year).max().to_numpy())


# ---------------------------------------------------------------- filters on the factor

def _factor_row(r: pd.Series) -> dict:
    """Filters 1, 2, 3 and 6 from one factor's daily returns up to the as-of date."""
    mon = v.monthly_table(r)
    row = dict(months=len(mon), phi=np.nan, f1=False, std_log_rv=np.nan, sort_ok=False, alpha=np.nan,
               t_alpha=np.nan, f3=False, alpha_e1=np.nan, alpha_e2=np.nan, alpha_e3=np.nan, f6=False)
    if len(mon) < SCREEN_RULES["min_months"]:
        return row
    t1 = v.t1_persistence(r, mon)
    span = v.t4_spanning(mon)
    t4 = v.t4_verdict(v.t4_sort(mon), span)
    eras = np.array_split(np.arange(len(mon)), SCREEN_RULES["n_eras"])
    era_alpha = [v.t4_spanning(mon.iloc[i])["alpha"] for i in eras]
    row.update(phi=t1["phi"], f1=t1["pass"],                        # F1 predictable risk
               std_log_rv=mon["log_rv"].std(),                      # F2 risk that moves (ranked later)
               sort_ok=t4["sort_ok"], alpha=span["alpha"], t_alpha=span["t_alpha"], f3=t4["pass"],  # F3
               f6=all(a > 0 for a in era_alpha))                    # F6 stable: alpha > 0 in every era
    row.update({f"alpha_e{k + 1}": a for k, a in enumerate(era_alpha)})
    return row


# ---------------------------------------------------------------- filters on the ETF

def _etf_row(factor: str, p: pd.DataFrame, o: pd.DataFrame) -> dict:
    """Filters 4 and 5 for the ETF standing in for `factor`. p, o: data up to the as-of date."""
    row = dict(etf=ETF_FOR.get(factor, "none"), etf_days=0, r2=np.nan, r2_mkt=np.nan, f4=False,
               dollar_vol=np.nan, f5=False)
    if factor not in ETF_FOR:
        return row
    y = p[ETF_FOR[factor].lower()].dropna()
    row["etf_days"] = len(y)
    if len(y) < SCREEN_RULES["etf_min_days"]:
        return row
    X = p.loc[y.index, ETF_REGRESSORS]
    row["r2"] = v.fit(y, X).rsquared
    row["r2_mkt"] = v.fit(y, X["mkt"]).rsquared                     # information only
    row["f4"] = bool(row["r2"] > SCREEN_RULES["etf_r2"])
    px = o[o["ticker"] == row["etf"]].tail(SCREEN_RULES["liquidity_days"])
    row["dollar_vol"] = (px["close"] * px["volume"]).median()
    row["f5"] = bool(row["dollar_vol"] > SCREEN_RULES["dollar_volume"])
    return row


# ---------------------------------------------------------------- the screen

def _reasons(r: pd.Series) -> str:
    """Every filter the factor fails, in filter order."""
    if r["months"] < SCREEN_RULES["min_months"]:
        out = [f"history: {r['months']} months, needs {SCREEN_RULES['min_months']}"]
    else:
        out = []
        if not r["f1"]:
            out.append(f"F1 predictable risk: phi {r['phi']:.2f}, needs > {v.RULES['t1_phi']} and Ljung-Box < 1%")
        if not r["f3"]:
            out.append(f"F3 exploitable link: alpha {r['alpha']:+.1%} (t {r['t_alpha']:.1f}), "
                       f"reward/risk {'falls' if r['sort_ok'] else 'does not fall steadily'} with risk")
        if not r["f6"]:
            out.append(f"F6 stable: alpha by era {r['alpha_e1']:+.1%} / {r['alpha_e2']:+.1%} / {r['alpha_e3']:+.1%}")
    if r["etf"] == "none":
        out.append("F4/F5 tradeable: no ETF in the data")
    elif r["etf_days"] < SCREEN_RULES["etf_min_days"]:
        out.append(f"F4/F5 tradeable: {r['etf']} has {r['etf_days']} days of history, needs {SCREEN_RULES['etf_min_days']}")
    else:
        if not r["f4"]:
            out.append(f"F4 systematic: {r['etf']} R² {r['r2']:.2f}, needs > {SCREEN_RULES['etf_r2']}")
        if not r["f5"]:
            out.append(f"F5 liquid: {r['etf']} dollar volume ${r['dollar_vol'] / 1e6:,.0f}M, "
                       f"needs > ${SCREEN_RULES['dollar_volume'] / 1e6:,.0f}M")
    return "; ".join(out)


def screen(panel: pd.DataFrame, ohlc: pd.DataFrame, asof) -> pd.DataFrame:
    """Screen every candidate factor as of `asof`, reading only data up to that date.

    panel: output of load_dev() (daily factor and ETF excess returns)
    ohlc:  output of load_etf_ohlc_dev() (long table with date, ticker, close, volume)
    asof:  the last trading day of a month, so no partial month is read as complete
    """
    asof = pd.Timestamp(asof)
    later = panel.index[panel.index > asof]
    if len(later) and later[0].to_period("M") == asof.to_period("M"):
        raise ValueError(f"{asof.date()} is mid-month; screen at a month's last trading day")
    p, o = panel.loc[:asof], ohlc[ohlc["date"] <= asof]
    df = pd.DataFrame([{"factor": f, **_factor_row(p[f]), **_etf_row(f, p, o)} for f in FACTORS]).set_index("factor")
    df["f2_rank"] = df["std_log_rv"].rank(ascending=False)             # 1 = most variable risk
    df["survivor"] = df[["f1", "f3", "f6", "f4", "f5"]].all(axis=1) & (df["months"] >= SCREEN_RULES["min_months"])
    df["cut_reason"] = [_reasons(r) for _, r in df.iterrows()]
    return df
