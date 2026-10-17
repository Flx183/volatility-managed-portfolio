import numpy as np
import pandas as pd
import pytest

from src import screen as sc
from src.config import FACTORS


def _data(seed=0, days=3800, volume=5e8):
    """Synthetic panel (7 factors + 4 ETFs, ~15 years) and ETF price table."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=days)
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, days)) * 0.3) * 0.01
    panel = pd.DataFrame(rng.normal(0, 1, (days, 11)) * vol[:, None] * 0.7,
                         index=idx.rename("date"), columns=FACTORS + ["spy", "mtum", "qual", "usmv"])
    panel["spy"] = panel["mkt"] * 1.0 + rng.normal(0, 0.0005, days)
    ohlc = pd.DataFrame([{"date": d, "ticker": t, "close": 100.0, "volume": volume}
                         for t in sc.ETF_FOR.values() for d in idx])
    return panel, ohlc


def test_nothing_after_the_asof_date_can_change_the_result():
    panel, ohlc = _data()
    asof = sc.year_ends(panel.index)[10]
    clean = sc.screen(panel, ohlc, asof)

    rng = np.random.default_rng(1)
    bad_p, bad_o = panel.copy(), ohlc.copy()
    bad_p.loc[asof + pd.Timedelta(days=1):] = rng.normal(0, 0.2, bad_p.loc[asof + pd.Timedelta(days=1):].shape)
    bad_o.loc[bad_o["date"] > asof, ["close", "volume"]] = [1e-3, 1.0]
    pd.testing.assert_frame_equal(clean, sc.screen(bad_p, bad_o, asof))
    pd.testing.assert_frame_equal(clean, sc.screen(panel.loc[:asof], ohlc[ohlc["date"] <= asof], asof))


def test_mid_month_asof_is_refused():
    panel, ohlc = _data()
    with pytest.raises(ValueError, match="mid-month"):
        sc.screen(panel, ohlc, panel.index[900])
    assert sc.screen(panel.loc[:panel.index[899]], ohlc, panel.index[899]) is not None   # frame ends at asof: allowed


def test_too_little_history_is_a_cut_not_a_pass():
    panel, ohlc = _data()
    asof = sc.year_ends(panel.index)[5]                      # ~6 years: below 120 months
    out = sc.screen(panel, ohlc, asof)
    assert not out["survivor"].any()
    assert out["cut_reason"].str.startswith("history:").all()


def test_etf_filters_and_factors_without_an_etf():
    panel, ohlc = _data(volume=5e8)                          # $50 billion a day
    asof = sc.year_ends(panel.index)[12]
    out = sc.screen(panel, ohlc, asof)
    assert out.loc["mkt", "f4"] and out.loc["mkt", "f5"]     # SPY tracks mkt, trades heavily
    assert out.loc["smb", "etf"] == "none" and not out.loc["smb", "survivor"]
    assert "no ETF" in out.loc["smb", "cut_reason"]

    thin = sc.screen(panel, ohlc.assign(volume=1e5), asof)   # $10 million a day
    assert not thin.loc["mkt", "f5"] and "F5 liquid" in thin.loc["mkt", "cut_reason"]


def test_survivor_needs_every_filter():
    panel, ohlc = _data()
    out = sc.screen(panel, ohlc, sc.year_ends(panel.index)[12])
    flags = out[["f1", "f3", "f6", "f4", "f5"]].all(axis=1) & (out["months"] >= 120)
    assert (out["survivor"] == flags).all()
    assert (out.loc[out["survivor"], "cut_reason"] == "").all()
