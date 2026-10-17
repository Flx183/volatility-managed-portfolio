import numpy as np
import pandas as pd
import pytest

from src import portfolio as pf
from src import variance as v


def _panel(seed=0, days=4000):
    """Synthetic panel, ~16 years: three factors with moving variance, SPY tracking the market
    and an ETF that is mostly market plus a momentum tilt."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=days, name="date")
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, days)) * 0.3) * 0.01
    z = rng.normal(0, 1, (days, 3)) * vol[:, None]
    panel = pd.DataFrame({"mkt": z[:, 0], "mom": 1.5 * z[:, 1] + 0.0002, "bab": 0.7 * z[:, 2] + 0.0001}, index=idx)
    panel["spy"] = panel["mkt"] + rng.normal(0, 0.0005, days)
    panel["mtum"] = 1.05 * panel["spy"] + 0.3 * panel["mom"] + rng.normal(0, 0.001, days)
    return panel


def test_nothing_after_a_date_changes_what_was_known_on_it():
    panel = _panel()
    cut = pd.Timestamp("2013-06-14")
    rec = pf.recipes(panel, "paper")
    out = pf.returns(panel, "paper", rec)

    bad = panel.copy()
    rng = np.random.default_rng(1)
    bad.loc[bad.index > cut] = rng.normal(0, 0.2, bad.loc[bad.index > cut].shape)
    bad_rec = pf.recipes(bad, "paper")
    known = rec.index.get_level_values("asof") <= cut
    pd.testing.assert_frame_equal(rec[known], bad_rec[bad_rec.index.get_level_values("asof") <= cut])
    pd.testing.assert_frame_equal(out.loc[:cut], pf.returns(bad, "paper", bad_rec).loc[:cut])


def test_first_recipe_needs_ten_years_and_holds_for_the_next_calendar_year():
    panel = _panel()
    rec = pf.recipes(panel, "paper")
    first = rec.index.get_level_values("asof")[0]
    assert panel.loc[:first].index.to_period("M").nunique() >= 120
    out = pf.returns(panel, "paper", rec)
    assert out.index[0].year == first.year + 1
    assert set(out.columns.get_level_values(0)) == set(pf.METHODS)


def test_weights_are_positive_sum_to_one_and_inverse_vol_equalises_risk():
    panel = _panel()
    x = panel[["mkt", "mom", "bab"]]
    for m in pf.METHODS:
        w = pf.mix_weights(x, m)
        assert (w >= 0).all() and w.sum() == pytest.approx(1)
    w = pf.mix_weights(x, "inv_vol")
    risk = w ** 2 * x.var()                                   # each piece's variance, ignoring correlation
    assert risk.max() / risk.min() == pytest.approx(1)


def test_inverse_vol_ignores_a_factors_scale_but_equal_and_inverse_variance_do_not():
    x = _panel()[["mkt", "mom", "bab"]]
    lev = x.assign(mom=2 * x["mom"])                          # the same bet, levered 2x
    for m, same in (("inv_vol", True), ("equal", False), ("inv_var", False)):
        risk, risk_lev = pf.mix_weights(x, m) * x.std(), pf.mix_weights(lev, m) * lev.std()
        share, share_lev = risk / risk.sum(), risk_lev / risk_lev.sum()
        assert np.allclose(share, share_lev) == same, m


def test_mean_variance_has_no_negative_weights_and_falls_back_to_equal():
    x = _panel()[["mkt", "mom", "bab"]]
    neg = x.assign(mkt=x["mkt"] - 0.002)                      # mkt mean far below zero
    w = pf.mix_weights(neg, "shrunk_mv")
    assert w["mkt"] == pytest.approx(1 / 6)                    # 0 from mean-variance, averaged with 1/3
    all_neg = x - 0.01
    assert np.allclose(pf.mix_weights(all_neg, "shrunk_mv"), 1 / 3)


def test_duplicates_drop_the_shorter_history_including_crisis_only_duplicates():
    rng = np.random.default_rng(3)
    months = pd.period_range("1990-01", periods=200, freq="M")
    rv = pd.Series(rng.lognormal(-6, 0.8, 200), index=months)
    a = pd.Series(rng.normal(0, 1, 200), index=months)
    crisis = rv >= rv.quantile(0.9)
    b = pd.Series(rng.normal(0, 1, 200), index=months).where(~crisis, a + rng.normal(0, 0.1, 200))
    mon = pd.DataFrame({"a": a, "b": b, "c": rng.normal(0, 1, 200)}, index=months)
    full, cri, n = pf.correlations(mon, rv)
    assert abs(full.loc["a", "b"]) < 0.7 < cri.loc["a", "b"] and n == crisis.sum()
    dropped = pf.dedupe(mon, rv, {"a": 500, "b": 300, "c": 500}, strength=lambda k: 0.0)
    assert list(dropped) == ["b"] and "with a" in dropped["b"]
    tie = pf.dedupe(mon, rv, {"a": 300, "b": 300, "c": 300}, strength={"a": 1.0, "b": 2.0, "c": 0.0}.get)
    assert list(tie) == ["a"]                                  # equal history: the weaker Test 4 t-stat goes
    assert pf.dedupe(mon.iloc[-60:], rv, {"a": 500, "b": 300, "c": 500}, lambda k: 0.0) == {}   # too short to judge


def test_a_strongly_negative_pair_is_a_hedge_not_a_duplicate():
    rng = np.random.default_rng(4)
    months = pd.period_range("1990-01", periods=200, freq="M")
    rv = pd.Series(rng.lognormal(-6, 0.8, 200), index=months)
    a = pd.Series(rng.normal(0, 1, 200), index=months)
    mon = pd.DataFrame({"a": a, "b": -a + rng.normal(0, 0.2, 200)})
    full, crisis, _ = pf.correlations(mon, rv)
    assert full.loc["a", "b"] < -0.9 and crisis.loc["a", "b"] < -0.9
    assert pf.dedupe(mon, rv, {"a": 500, "b": 300}, lambda k: 0.0) == {}


def test_etf_track_trades_the_tilt_and_positions_reproduce_p():
    panel = _panel()
    rec = pf.recipes(panel, "etf")
    out = pf.returns(panel, "etf", rec)
    asof = rec.index.get_level_values("asof").unique()[3]
    row = rec.loc[(asof, "inv_vol")]
    assert row["pos_spy"] == pytest.approx(row["w_spy"] - row["w_mtum"] * row["beta_mtum"])
    year = panel.loc[panel.index.year == asof.year + 1]
    p = year["spy"] * row["pos_spy"] + year["mtum"] * row["pos_mtum"]
    assert np.allclose(out.loc[year.index, ("inv_vol", "p")], p)
    assert np.allclose(out[("inv_vol", "managed")], out[("inv_vol", "scale")] * out[("inv_vol", "p")])


def test_serving_size_is_c_over_last_months_variance():
    panel = _panel()
    rec = pf.recipes(panel, "paper")
    out = pf.returns(panel, "paper", rec, max_size=None)
    asof = rec.index.get_level_values("asof").unique()[2]
    c = rec.loc[(asof, "equal"), "c"]
    p = panel[["mkt", "mom", "bab"]].mean(axis=1)              # equal weight: p is the plain average
    rv = v.monthly_table(p.loc[asof.replace(day=1):])["rv"]
    march = out.loc[str(asof.year + 1) + "-03", ("equal", "scale")]
    assert np.allclose(march, c / rv.loc[f"{asof.year + 1}-02"])
    jan = out.loc[str(asof.year + 1) + "-01", ("equal", "scale")]
    assert np.allclose(jan, c / rv.loc[f"{asof.year}-12"])     # January: December under the new recipe


def test_serving_size_is_capped():
    panel = _panel()
    rec = pf.recipes(panel, "paper")
    free = pf.returns(panel, "paper", rec, max_size=None)
    capped = pf.returns(panel, "paper", rec)                   # default: the primary test's cap
    cap = pf.PORTFOLIO_RULES["max_size"]
    for m in pf.METHODS:
        assert free[(m, "scale")].max() > cap                  # the cap binds somewhere in this sample
        assert np.allclose(capped[(m, "scale")], free[(m, "scale")].clip(upper=cap))
        assert np.allclose(capped[(m, "managed")], capped[(m, "scale")] * capped[(m, "p")])
        assert np.allclose(capped[(m, "p")], free[(m, "p")])   # the cap never touches p itself


def test_tests_on_p_are_not_judged_on_a_short_series():
    panel = _panel()
    short = pf.tests(panel["mkt"].iloc[-1000:])
    assert short["months"] < 120 and short["judged"] is False and short["pass"] is None
    full = pf.tests(panel["mkt"])
    assert full["judged"] and isinstance(full["pass"], bool)
