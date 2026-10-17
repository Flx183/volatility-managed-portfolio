"""Unit tests for each engine stage alone, then for the pipeline. The five engine checks of the
plan (worlds, constant, shuffled, oracle, hand check) are in test_engine_checks.py."""

import numpy as np
import pandas as pd
import pytest

from src import engine as e
from src import portfolio as pf
from src.engine import data, evaluation as ev, execution as ex, forecast as fc, synthetic as syn, weights as wt


def _world(years=15, seed=0, world="flat"):
    d = syn.garch(years=years, world=world, seed=seed, start="1990-01-01")
    return d["x"], d["rf"]


def _two_asset_book(years=16, seed=0):
    """Two instruments and a recipe that changes every year; the second year's recipe shorts A."""
    a, rf = _world(years, seed)
    b, _ = _world(years, seed + 1)
    months = a.index.to_period("M").unique()
    rng = np.random.default_rng(seed)
    by_year = {y: (rng.uniform(-0.3, 1.0), rng.uniform(0.3, 1.0)) for y in months.year.unique()}
    pos = pd.DataFrame([by_year[y] for y in months.year], index=months, columns=["A", "B"])
    return e.Book(pd.DataFrame({"A": a, "B": b}), rf, pos)


# ---------------------------------------------------------------- stage 1: data

def test_book_rejects_bad_inputs():
    x, rf = _world(2)
    months = x.index.to_period("M").unique()
    with pytest.raises(ValueError, match="no returns"):
        e.Book(x.to_frame("A"), rf, pd.DataFrame(1.0, index=months, columns=["B"]))
    with pytest.raises(ValueError, match="monthly periods"):
        e.Book(x.to_frame("A"), rf, pd.DataFrame(1.0, index=months.to_timestamp(), columns=["A"]))
    with pytest.raises(ValueError, match="risk-free"):
        e.Book(x.to_frame("A"), rf.iloc[10:], pd.DataFrame(1.0, index=months, columns=["A"]))


def test_recipe_ids_group_equal_positions():
    pos = pd.DataFrame({"A": [1, 1, 0.5, 0.5, 1], "B": [0, 0, 0.5, 0.5, 0]},
                       index=pd.period_range("2000-01", periods=5, freq="M"))
    assert data.recipe_ids(pos).tolist() == [0, 0, 1, 1, 0]


def test_monthly_return_is_compounded_total_minus_compounded_rf():
    x, rf = _world(3)
    mon = data.monthly(x, rf)
    jan = x.loc["1991-01"]
    total, cash = np.prod(1 + jan + rf.loc["1991-01"]), np.prod(1 + rf.loc["1991-01"])
    assert mon.loc["1991-01", "ret"] == pytest.approx(total - cash, rel=1e-12)
    assert mon.loc["1991-01", "rf"] == pytest.approx(cash - 1, rel=1e-12)
    assert mon.loc["1991-01", "rv"] == pytest.approx((jan ** 2).sum(), rel=1e-12)


def test_mix_is_the_position_weighted_sum_on_days_all_instruments_exist():
    book = _two_asset_book(3)
    pos = pd.Series({"A": 0.4, "B": -0.2})
    r = book.returns.copy()
    r.iloc[:5, 1] = np.nan                                       # B starts later
    p = data.mix(e.Book(r, book.rf, book.positions), pos)
    assert p.index[0] == r.index[5]
    assert np.allclose(p, 0.4 * r["A"].iloc[5:] - 0.2 * r["B"].iloc[5:])


def test_phase4_recipes_become_positions_held_the_next_calendar_year():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2000-01-03", periods=4000, name="date")
    panel = pd.DataFrame(rng.normal(0, 0.01, (4000, 3)), index=idx, columns=["mkt", "mom", "bab"])
    panel["rf"] = 0.0001
    rec = pf.recipes(panel, "paper")
    book = data.from_phase4(panel, rec, "inv_vol")
    first = rec.index.get_level_values("asof")[0]
    assert book.positions.index[0] == pd.Period(f"{first.year + 1}-01", "M")
    row = rec.loc[(first, "inv_vol")]
    assert np.allclose(book.positions.loc[f"{first.year + 1}-07"], [row["pos_mkt"], row["pos_mom"], row["pos_bab"]])


# ---------------------------------------------------------------- stage 2: forecast

def test_forecasters_measure_what_they_say():
    x, _ = _world(3)
    jan = x.loc["1991-01"]
    assert fc.last_rv(x).loc["1991-01"] == pytest.approx((jan ** 2).sum())
    assert fc.last_rv_down(x).loc["1991-01"] == pytest.approx((jan.clip(upper=0) ** 2).sum())
    upto = x.loc[:"1991-01"]
    assert fc.rv_days(126)(x).loc["1991-01"] == pytest.approx(21 * (upto.iloc[-126:] ** 2).mean())
    assert (fc.constant(0.002)(x) == 0.002).all()
    assert fc.oracle(x).loc["1991-01"] == pytest.approx((x.loc["1991-02"] ** 2).sum())


@pytest.mark.parametrize("forecaster", [fc.last_rv, fc.last_rv_down, fc.rv_days(126), fc.rv_days(22)])
def test_forecasts_use_only_data_up_to_their_month(forecaster):
    x, _ = _world(5)
    cut = pd.Timestamp("1992-06-30")
    noisy = x.copy()
    noisy[noisy.index > cut] = np.random.default_rng(1).normal(0, 0.05, (noisy.index > cut).sum())
    a, b = forecaster(x), forecaster(noisy)
    assert np.allclose(a.loc[:"1992-06"], b.loc[:"1992-06"], equal_nan=True)
    assert not np.allclose(a.loc["1992-07":], b.loc["1992-07":])


def test_oracle_looks_ahead_and_says_so():
    x, _ = _world(5)
    noisy = x.copy()
    noisy.loc["1992-07"] = 0.05
    assert fc.oracle(x).loc["1992-06"] != fc.oracle(noisy).loc["1992-06"]
    assert getattr(fc.oracle, "looks_ahead", False)


def test_permute_reorders_the_same_values():
    f = fc.last_rv(_world(5)[0])
    p = fc.permute(f, seed=3)
    assert np.allclose(np.sort(p), np.sort(f)) and not np.allclose(p, f)
    assert p.index.equals(f.index)


# ---------------------------------------------------------------- stage 3: weights

def test_signal_powers_and_positive_forecasts():
    f = pd.Series([0.01, 0.04], index=pd.period_range("2000-01", periods=2, freq="M"))
    assert np.allclose(wt.signal(f, 1.0), [100, 25])            # volatility doubles: size / 4
    assert np.allclose(wt.signal(f, 0.5), [10, 5])              # volatility doubles: size / 2
    with pytest.raises(ValueError, match="positive"):
        wt.signal(f - 0.02, 1.0)


def test_realtime_c_matches_its_formula_by_brute_force():
    x, rf = _world(15)
    mon = data.monthly(x, rf)
    s = wt.signal(fc.last_rv(x), 1.0)
    c = wt.realtime_c(mon["ret"], s, min_months=120)
    months = mon.index
    for t in [months[120], months[121], months[150], months[-1]]:
        u = months[(months < t)]                                 # pairs s with s+1 <= t
        u = u[u + 1 <= t]
        y = mon["ret"].reindex(u + 1).to_numpy()
        z = y * s.reindex(u).to_numpy()
        assert c.loc[t] == pytest.approx(np.std(y, ddof=1) / np.std(z, ddof=1), rel=1e-12)
    assert c.loc[:months[119]].isna().all() and c.loc[months[120]:].notna().all()


def test_realtime_c_ignores_the_return_of_the_month_after():
    x, rf = _world(15)
    mon = data.monthly(x, rf)
    s = wt.signal(fc.last_rv(x), 1.0)
    t = mon.index[150]
    bumped = mon["ret"].copy()
    bumped.loc[t + 1] += 0.5
    c0, c1 = wt.realtime_c(mon["ret"], s), wt.realtime_c(bumped, s)
    assert c0.loc[t] == c1.loc[t] and c0.loc[t + 1] != c1.loc[t + 1]


def test_constant_signal_gives_size_one_and_window_option_works():
    x, rf = _world(15)
    mon = data.monthly(x, rf)
    s = pd.Series(500.0, mon.index)
    c = wt.realtime_c(mon["ret"], s)
    assert np.allclose((c * s).dropna(), 1.0)
    roll = wt.realtime_c(mon["ret"], wt.signal(fc.last_rv(x), 1.0), min_months=60, window=60)
    assert roll.notna().sum() == len(mon) - 60


def test_size_is_capped_and_floored_and_target_c_hits_the_target():
    sig = pd.Series([1.0, 2.0, 3.0], index=pd.period_range("2000-01", periods=3, freq="M"))
    assert wt.size(0.6, sig, 1.5).tolist() == [0.6, 1.2, 1.5]
    assert wt.size(-1.0, sig, 1.5).tolist() == [0, 0, 0]
    assert wt.size(0.6, sig, None).tolist() == pytest.approx([0.6, 1.2, 1.8])
    monthly_var = 0.12 ** 2 / 12                                 # forecast exactly at the 12% target
    assert wt.target_c(0.12) * monthly_var ** -0.5 == pytest.approx(1.0)


# ---------------------------------------------------------------- stage 4: execution

def test_period_labels_follow_the_trade_dates():
    idx = pd.bdate_range("2001-01-01", "2001-04-30")
    assert (ex.period_labels(idx, 0) == idx.to_period("M")).all()
    late = ex.period_labels(idx, 1)
    first_feb = idx[idx.month == 2][0]
    assert late[idx.get_loc(first_feb)] == pd.Period("2001-01", "M")          # traded a day after month-end
    assert (late[idx.month == 3][1:] == pd.Period("2001-03", "M")).all()


def _periods_and_returns(costs=ex.NO_COSTS):
    x, rf = _world(3)
    book = e.Book.single(x, rf, "A")
    per = ex.holding_periods(book, costs)
    months = per.growth.index[1:13]
    return per, months, per.growth.loc[months, "A"] - 1, per.cash.loc[months] - 1


def test_holding_one_unit_earns_the_asset_and_trades_only_at_entry():
    per, months, R, Rf = _periods_and_returns()
    out = ex.simulate(per, pd.DataFrame({"A": 1.0}, index=months))
    assert np.allclose(out["net"], R) and np.allclose(out["excess"], R - Rf)
    assert out["turnover"].iloc[0] == 1.0 and np.allclose(out["turnover"].iloc[1:], 0)


def test_drift_and_trading_cost_follow_the_plan_formulas():
    costs = ex.Costs(kappa={"A": 0.001})
    per, months, R, Rf = _periods_and_returns(costs)
    w = pd.Series(np.linspace(0.3, 1.4, len(months)), months)
    out = ex.simulate(per, w.to_frame("A"))
    Rp = w * R + (1 - w) * Rf                                    # R_p = w R + (1 - w) R_f
    drift = w * (1 + R) / (1 + Rp)                               # w̃ = w (1 + R) / (1 + R_p)
    assert np.allclose(out["pre_trade"], Rp) and np.allclose(out["drift_A"], drift)
    trade = (w - drift.shift(1).fillna(0)).abs()
    assert np.allclose(out["turnover"], trade) and np.allclose(out["trading"], 0.001 * trade)
    assert np.allclose(out["net"], (1 - 0.001 * trade) * (1 + Rp) - 1)


def test_borrowing_and_shorting_pay_their_fees():
    x, rf = _world(3)
    costs = ex.Costs(borrow_spread=0.005, short_fee={"A": 0.0025})
    per = ex.holding_periods(e.Book.single(x, rf, "A"), costs)
    months = per.growth.index[1:13]
    long = ex.simulate(per, pd.DataFrame({"A": 1.5}, index=months))       # borrows half the account
    short = ex.simulate(per, pd.DataFrame({"A": -0.5}, index=months))     # shorts half the account
    free = ex.holding_periods(e.Book.single(x, rf, "A"))
    assert np.allclose(long["gross"], ex.simulate(free, pd.DataFrame({"A": 1.5}, index=months))["net"])
    # the same charges from the daily data: daily accrual, 252 days a year
    g = x.groupby(x.index.to_period("M"))
    borrow = 0.5 * ((1 + rf + 0.005 / 252).groupby(g.ngroup().to_numpy()).prod().to_numpy()
                    - (1 + rf).groupby(g.ngroup().to_numpy()).prod().to_numpy())
    fee = 0.5 * ((1 + x + rf + 0.0025 / 252).groupby(x.index.to_period("M")).prod()
                 - (1 + x + rf).groupby(x.index.to_period("M")).prod())
    month_pos = per.growth.index.get_indexer(months)
    assert np.allclose(long["financing"], borrow[month_pos], rtol=1e-10)
    assert np.allclose(short["financing"], fee.loc[months], rtol=1e-10)
    assert (long["financing"] > 0).all() and (short["financing"] > 0).all()


def test_a_mix_held_at_fixed_positions_earns_its_weighted_return():
    book = _two_asset_book(3)
    per = ex.holding_periods(book)
    months = per.growth.index[1:10]
    w = pd.DataFrame({"A": -0.3, "B": 0.9}, index=months)
    out = ex.simulate(per, w)
    R = per.growth.loc[months] - 1
    Rf = per.cash.loc[months] - 1
    assert np.allclose(out["net"], -0.3 * R["A"] + 0.9 * R["B"] + 0.4 * Rf)


def test_simulate_refuses_gaps_and_missing_returns():
    per, months, _, _ = _periods_and_returns()
    with pytest.raises(ValueError, match="consecutive"):
        ex.simulate(per, pd.DataFrame({"A": 1.0}, index=months.delete(3)))
    x, rf = _world(3)
    x.loc["1990-03-07"] = np.nan
    per = ex.holding_periods(e.Book.single(x, rf, "A"))
    with pytest.raises(ValueError, match="no return in 1990-03"):
        ex.simulate(per, pd.DataFrame({"A": 1.0}, index=pd.period_range("1990-02", periods=3, freq="M")))


# ---------------------------------------------------------------- stage 5: evaluation

def test_spanning_recovers_a_known_alpha_and_beta():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.005, 0.04, 600))
    y = 0.002 + 0.8 * x + rng.normal(0, 0.01, 600)
    sp = ev.spanning(y, x)
    assert sp["alpha"] == pytest.approx(0.024, abs=0.006) and sp["beta"] == pytest.approx(0.8, abs=0.03)
    assert ev.alpha(y, x) == pytest.approx(sp["alpha"], rel=1e-10)
    assert ev.sharpe(x) == pytest.approx(x.mean() / x.std() * np.sqrt(12))


def test_break_even_cost_equalises_the_sharpe_ratios():
    d = syn.garch(years=200, world="flat", seed=0)
    res = e.run(e.Book.single(d["x"], d["rf"]))
    k = ev.break_even(res)
    assert 0 < k < np.inf
    assert ev.sharpe(ev.excess_at(res, k)) == pytest.approx(ev.sharpe(ev.excess_at(res, k, "bench_")), abs=1e-8)
    z = ev.zero_profit_cost(res)
    assert ev.excess_at(res, z).mean() == pytest.approx(0, abs=1e-12) and z > k
    assert ev.placebo_share(0.5, np.array([0.1, 0.4, 0.6, 0.9])) == 0.5


def _pair(n, rng, sr2=0.125, rho=0.6):
    """Monthly excess returns of two series: Sharpe 0.125 and sr2 a month, correlation rho,
    volatilities 4% and 6%, as an engine result frame."""
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
    return pd.DataFrame({"excess": 0.005 + 0.04 * z1, "bench_excess": sr2 * 0.06 + 0.06 * z2, "size": 1.0})


def test_sharpe_gain_standard_error_matches_the_normal_formula():
    rng = np.random.default_rng(0)
    n, rho, s1, s2 = 200_000, 0.6, 0.125, 0.2
    t = ev.sharpe_gain_test(_pair(n, rng, s2, rho), lags=0)
    memmel = np.sqrt((2 * (1 - rho) + 0.5 * (s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * rho ** 2)) / n) * np.sqrt(12)
    assert t["se"] == pytest.approx(memmel, rel=0.02)              # Jobson-Korkie / Memmel, normal iid
    assert t["gain"] == pytest.approx((s1 - s2) * np.sqrt(12), abs=4 * memmel)
    assert t["mde"] == pytest.approx((1.6448536 + 0.8416212) * t["se"], rel=1e-6)


def test_sharpe_gain_test_rejects_five_percent_of_the_time_when_there_is_no_gain():
    rng = np.random.default_rng(1)
    p = np.array([ev.sharpe_gain_test(_pair(600, rng))["p"] for _ in range(2000)])   # equal Sharpe ratios
    assert 0.03 < (p < 0.05).mean() < 0.075


def test_the_three_readings_in_their_order():
    assert ev.reading({"p": 0.01, "upper": 0.05}) == "edge"                     # rejection comes first
    assert ev.reading({"p": 0.30, "upper": 0.08}) == "no meaningful edge"
    assert ev.reading({"p": 0.30, "upper": 0.25}) == "inconclusive"
    t = ev.sharpe_gain_test(_pair(600, np.random.default_rng(2)))
    assert t["lower"] < t["gain"] < t["upper"] and t["reading"] in ("edge", "no meaningful edge", "inconclusive")
    assert {"volatility: timed", "volatility: buy-and-hold", "mean size"} <= set(t)


def test_summary_reports_realized_volatility_and_size_for_both_runs():
    x, rf = _world(30)
    s = ev.summary(e.run(e.Book.single(x, rf)))
    assert s["volatility"].notna().all() and set(s.index) == {"timed", "buy-and-hold"}
    assert s.loc["timed", "mean_size"] > 0


def test_the_short_fee_stress_case_changes_only_the_fee():
    base, stress = ex.COSTS["etf"], ex.COSTS["etf_short_stress"]
    assert base.short_fee["spy"] == 0.0025 and stress.short_fee["spy"] == 0.01
    assert base.kappa == stress.kappa and base.borrow_spread == stress.borrow_spread
    assert e.ENGINE_RULES["etf_min_months"] == 36


def test_no_timing_gain_means_a_break_even_of_zero():
    d = syn.garch(years=30, world="flat", seed=0)
    res = e.run(e.Book.single(d["x"], d["rf"]), forecaster=fc.constant(0.003))   # size 1: no gain
    assert ev.break_even(res.assign(excess=res["bench_excess"] - 1e-6)) == 0.0


# ---------------------------------------------------------------- the pipeline

def test_nothing_after_a_date_changes_what_was_decided_or_earned_before_it():
    book = _two_asset_book(16)
    cut = pd.Timestamp("2003-06-17")
    res = e.run(book, min_months=60)
    r = book.returns.copy()
    r.loc[r.index > cut] = np.random.default_rng(9).normal(0, 0.05, r.loc[r.index > cut].shape)
    bad = e.run(e.Book(r, book.rf, book.positions), min_months=60)
    pd.testing.assert_frame_equal(res.loc[:"2003-05"], bad.loc[:"2003-05"])
    decision = ["recipe", "forecast", "signal", "c", "size", "capped"]
    pd.testing.assert_frame_equal(res.loc["2003-06":"2003-06", decision], bad.loc["2003-06":"2003-06", decision])
    assert not np.isclose(res.loc["2003-06", "net"], bad.loc["2003-06", "net"])


def test_january_is_sized_on_the_new_recipe_applied_to_december():
    book = _two_asset_book(16)
    res = e.run(book, min_months=60)
    jan = pd.Period("2003-01", "M")
    p_new = data.mix(book, book.positions.loc[jan])               # the mix held in January
    assert res.loc[jan, "forecast"] == pytest.approx((p_new.loc["2002-12"] ** 2).sum(), rel=1e-12)
    assert res.loc[jan, "recipe"] != res.loc[jan - 1, "recipe"]


def test_buy_and_hold_is_the_same_book_at_size_one():
    book = _two_asset_book(16)
    res = e.run(book, min_months=60)
    assert np.allclose(res[["bench_w_A", "bench_w_B"]], book.positions.loc[res.index])
    assert np.allclose(res[["w_A", "w_B"]], book.positions.loc[res.index].mul(res["size"], axis=0))


def test_with_a_fixed_c_and_no_cap_the_sharpe_ratio_ignores_c():
    x, rf = _world(30)
    book = e.Book.single(x, rf)
    a = e.run(book, c=0.001, max_size=None)
    b = e.run(book, c=0.003, max_size=None)
    assert np.allclose(b["excess"], 3 * a["excess"])
    assert ev.sharpe(a["excess"]) == pytest.approx(ev.sharpe(b["excess"]), rel=1e-12)


def test_a_new_forecaster_plugs_in_and_nothing_else_changes():
    x, rf = _world(30)
    book = e.Book.single(x, rf)
    base, other = e.run(book), e.run(book, forecaster=fc.rv_days(126))
    assert list(base.columns) == list(other.columns)
    assert not np.allclose(base["forecast"], other["forecast"].reindex(base.index))
    assert np.allclose(other["bench_excess"].reindex(base.index).dropna(),
                       base["bench_excess"].reindex(other.index).dropna())


def test_rules_differ_only_in_how_hard_they_lean():
    x, rf = _world(30)
    book = e.Book.single(x, rf)
    var, vol = e.run(book, "inv_var", max_size=None), e.run(book, "inv_vol", max_size=None)
    assert var["size"].std() > vol["size"].std()                 # inverse variance swings harder
    down = e.run(book, "down_vol")
    assert np.allclose(down["forecast"], fc.last_rv_down(x).reindex(down.index - 1).to_numpy())


def test_a_missing_size_in_the_middle_is_an_error():
    x, rf = _world(20)
    book = e.Book.single(x, rf)
    holes = lambda p: fc.last_rv(p).drop(pd.Period("2005-03", "M"))
    with pytest.raises(ValueError, match="no size for holding month 2005-04"):
        e.run(book, forecaster=holes)
