"""The code review of Phase 8 (2026-10-19): one test per finding that reproduces its failure scenario, and
the invariants that guard the fixes. Simulated data only; no test touches the forecast cache (md.frozen
is replaced by cheap stand-ins). Finding 13 is in test_frozen_rules.py; finding 15 (stale cache) is
checked by rebuilding the cache from scratch (PREREGISTRATION Entry 12)."""

import numpy as np
import pandas as pd
import pytest

from src import engine as e
from src import models as md
from src.config import FACTORS
from src.engine import evaluation as ev, execution as ex, synthetic as syn
from src.engine.data import DAY, MONTH, WEEK
from src.models import filters, variations, weekly
from src.robustness import build, checks, crises, ledger
from src.walkforward import build as wfb, runs, screened


def _p(years=14, seed=0, start="1990-01-01"):
    return syn.garch(years=years, seed=seed, start=start)["x"].rename("p")


def _rf(x):
    return pd.Series(0.0001, x.index)


def _panel(years=28, seed=0, start="1980-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=252 * years, name="date")
    panel = pd.DataFrame(rng.normal(0.0003, 0.01, (len(idx), len(FACTORS))), index=idx, columns=FACTORS)
    panel["rf"] = 0.0001
    return panel


def _daily_rolling(p):
    p = p.dropna()
    days, _ = variations.daily_schedule(p)
    return pd.Series((p ** 2).rolling(22).sum().to_numpy()[days.to_numpy()], index=days.index, name="forecast")


def _fake_frozen(gjr, freq=MONTH, window=None):
    """md.frozen without GARCH or the cache: rolling RV at the frozen model's origins (monthly from the mix's
    120th month, weekly from the first week after it, daily from that day on)."""
    return {MONTH: filters.rolling, WEEK: weekly.rolling, DAY: _daily_rolling}[freq]


@pytest.fixture
def cheap(monkeypatch):
    monkeypatch.setattr(md, "frozen", _fake_frozen)


def _survivors(panel, held):
    """held(year) -> the list of factors kept at that year-end (held in the year after)."""
    return pd.Series({d: held(d.year) for d in screened.sc.year_ends(panel.index)})


ACCOUNT = ["gross", "financing", "pre_trade", "turnover", "trading", "net", "rf", "excess"]


def _accounts(res):
    w = [c for c in res.columns if c.startswith(("w_", "drift_", "bench_"))]
    return res[["size", "capped", "entry"] + ACCOUNT + w]


# ---------------------------------------------------------------- finding 1: the downside rule on daily books

def test_the_downside_rule_forecasts_a_daily_book_on_its_days():
    p = _p()
    out = runs.downside(_daily_rolling, DAY)(p)
    base = _daily_rolling(p)
    share = runs.down_share(p)
    assert out.index.equals(base.index) and out.notna().all()
    d = base.index[300]
    assert out[d] == pytest.approx(base[d] * share[d.to_timestamp()], rel=1e-12)


def test_the_downside_rule_refuses_a_frequency_it_does_not_know():
    with pytest.raises(ValueError, match="frequency"):
        runs.downside(filters.rolling, "Q")


# ---------------------------------------------------------------- finding 2: one change at a time

def test_phase_8_refuses_a_run_that_changes_two_things(cheap):
    panel = _panel()
    rec = checks.recipes(panel, "screened", _survivors(panel, lambda y: ["mkt"] if y >= 1999 else []))
    with pytest.raises(ValueError, match="one thing at a time"):
        checks.run(panel, rec, {"order": "each", "rule": "inv_vol"})
    with pytest.raises(ValueError, match="one thing at a time"):
        checks.variation(checks.BASE, order="each", window="rolling 10 years")
    with pytest.raises(ValueError, match="one thing at a time"):
        checks.resolve(checks.Spec(order="each", cap=None))


# ---------------------------------------------------------------- finding 3: an asset too short to size

def test_an_asset_too_short_to_size_leaves_its_segment_in_cash_in_both_orders(cheap):
    panel = _panel()                                                # 1980-2007
    panel.loc[:"1993", "rmw"] = np.nan                              # rmw from 1994: its c_t would need 2014
    held = lambda y: ["mkt"] if 2000 <= y <= 2002 else ["mkt", "rmw"] if y >= 2004 else []
    rec = checks.recipes(panel, "screened", _survivors(panel, held))
    _, comb, _ = checks.run(panel, rec, {})
    _, each, _ = checks.run(panel, rec, {"order": "each"})
    assert comb.index.equals(each.index) and comb.index[-1] == pd.Period("2003-12", MONTH)
    pd.testing.assert_frame_equal(_accounts(comb), _accounts(each), check_exact=True)    # one asset there


# ---------------------------------------------------------------- finding 4: weekly months are calendar months

def test_phase_8_scores_a_weekly_month_on_that_months_days():
    x = _p(24)
    book = e.Book.single(x, _rf(x), "A", freq=WEEK)
    per = ex.holding_periods(book)
    res = runs.run(runs.prepare(book, weekly.rolling), 1.0, per, min_pairs=520)
    m = checks.scored(book, res, per)
    straddled = [mo for mo in m.index if mo.to_timestamp(how="end").normalize() not in set(per.closes)]
    assert len(straddled) > 20                                      # months whose last trading day is no week's close
    for mo in straddled[:5] + straddled[-5:]:
        d = x.loc[str(mo)].index
        truth = np.prod(1 + x.loc[d] + _rf(x).loc[d]) - np.prod(1 + _rf(x).loc[d])
        assert m.loc[mo, "bench_excess"] == pytest.approx(truth, rel=1e-12)   # buy-and-hold holds A at 1, no cash
    # Phase 7's rule gives each week to the month of its closing trade: not the calendar month (kept for Phase 7)
    assert not np.allclose(runs.scored(res, per).loc[m.index, "bench_excess"], m["bench_excess"])


def test_a_weekly_month_moves_with_its_own_last_days_only():
    x = _p(24)
    book = e.Book.single(x, _rf(x), "A", freq=WEEK)
    per = ex.holding_periods(book)
    res = runs.run(runs.prepare(book, weekly.rolling), 1.0, per, min_pairs=520)
    m = checks.scored(book, res, per)
    mo = next(mo for mo in m.index[24:] if x.loc[str(mo)].index[-1].dayofweek < 4)    # ends Monday to Thursday
    last = x.loc[str(mo)].index[-1]
    y = x.copy()
    y.loc[last] += 0.05                                             # inside a week that closes next month
    b2 = e.Book.single(y, _rf(y), "A", freq=WEEK)
    p2 = ex.holding_periods(b2)
    r2 = res.copy()                                                 # same sizes: only the day's return differs
    pos = b2.positions.loc[res.index]
    r2[ex.simulate(p2, pos.mul(res["size"], axis=0)).columns] = ex.simulate(p2, pos.mul(res["size"], axis=0))
    r2[[f"bench_{c}" for c in ACCOUNT if c != "rf"]] = ex.simulate(p2, pos).drop(columns="rf")[[c for c in ACCOUNT if c != "rf"]].to_numpy()
    m2 = checks.scored(b2, r2, p2)
    assert m2.loc[mo, "excess"] != m.loc[mo, "excess"]
    assert np.array_equal(m2.loc[:mo - 1, "excess"], m.loc[:mo - 1, "excess"])


# ---------------------------------------------------------------- finding 5: one trading calendar per recipe

def _two_instruments(missing):
    """A daily book holding A (one unit) in 2003-2004, with B (never held) trading every day. A has no return
    on `missing` days; B does, so they stay days of book.returns."""
    x = _p(20, start="1985-01-01")
    rx = pd.DataFrame({"A": x, "B": x.shift(1).fillna(0.0)})
    rx.loc[pd.DatetimeIndex(missing), "A"] = np.nan
    days = rx.loc["2003":"2004"].index.to_period(DAY)
    return e.Book(rx, _rf(x), pd.DataFrame({"A": 1.0, "B": 0.0}, index=days))


def test_a_daily_decision_reads_the_recipes_own_calendar():
    first = pd.Timestamp("2003-01-01")
    book = _two_instruments(["2002-12-31"])                         # A misses the close before the first day
    dec = e.decide(book, e.prepare(book, _daily_rolling), 1.0, min_months=500)
    f = _daily_rolling(book.returns["A"])
    assert dec.index[0] == first.to_period(DAY)
    assert dec["forecast"].iloc[0] == f[pd.Period("2002-12-30", DAY)]       # A's last close before it


def test_a_daily_book_held_on_a_day_its_assets_have_no_return_stops_with_that_day():
    book = _two_instruments(["2003-06-10"])
    with pytest.raises(ValueError, match="2003-06-10.*no return"):
        e.decide(book, e.prepare(book, _daily_rolling), 1.0, min_months=500)


def test_a_daily_forecast_must_cover_its_mixs_calendar():
    book = _two_instruments([])
    gappy = lambda p: _daily_rolling(p).drop(pd.Period("2002-05-07", DAY))
    with pytest.raises(ValueError, match="calendar"):
        e.prepare(book, gappy)


# ---------------------------------------------------------------- finding 6: no decision is not a cap

def test_an_asset_without_a_decision_is_never_capped():
    panel = _panel()
    panel.loc[:"1990", "mom"] = np.nan                              # momentum has no size until 2006, the market from 1995
    book = checks.book(panel, checks.recipes(panel, "screened", _survivors(panel, lambda y: ["mkt", "mom"] if y >= 1999 else [])))
    cfg = checks.resolve(checks.BASE, frozen=_fake_frozen)
    S, C = checks.asset_sizes(book, cfg["forecaster"], cfg["power"], cfg["max_size"], 60, cfg["c_window"])
    assert (C.dtypes == bool).all() and S["mom"].isna().any() and not (C & S.isna()).any().any()
    assert not C.loc[S["mom"].isna(), "mom"].any()


# ---------------------------------------------------------------- finding 7: change in gain, split

def _monthly(n=300, seed=0, start="1980-01"):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.006, 0.04, n)
    return pd.DataFrame({"excess": 0.8 * b + rng.normal(0.001, 0.01, n), "bench_excess": b},
                        index=pd.period_range(start, periods=n, freq=MONTH))


def test_the_change_in_gain_splits_into_a_spec_effect_and_a_sample_effect():
    base = _monthly()
    common = base.index[24:]
    var = base.loc[common].assign(excess=lambda d: d["excess"] * 1.1 + 0.0005)
    gain = lambda m: ev.sharpe(m["excess"]) - ev.sharpe(m["bench_excess"])
    fx = checks.effects(base.loc[common], var, base)
    assert fx["spec_effect"] == pytest.approx(gain(var) - gain(base.loc[common]), abs=1e-15)
    assert fx["sample_effect"] == pytest.approx(gain(base.loc[common]) - gain(base), abs=1e-15)
    assert fx["spec_effect"] + fx["sample_effect"] == pytest.approx(gain(var) - gain(base), abs=1e-14)   # the old change_in_gain
    assert fx["spec_lower"] < fx["spec_effect"] < fx["spec_upper"] and fx["spec_se"] > 0
    with pytest.raises(ValueError, match="same months"):
        checks.effects(base, var, base)


def test_an_identity_variation_changes_nothing_exactly(cheap):
    assert checks.variation(checks.BASE, cap=checks.BASE.cap) == checks.BASE
    base = _monthly()
    fx = checks.effects(base, base.copy(), base)
    assert fx["spec_effect"] == 0.0 and fx["spec_se"] == 0.0 and fx["sample_effect"] == 0.0
    panel = _panel()
    rec = checks.recipes(panel, "screened", _survivors(panel, lambda y: ["mkt", "bab"] if y >= 1999 else []))
    a = checks.run(panel, rec, checks.BASE)
    b = checks.run(panel, rec, checks.variation(checks.BASE, cap=checks.BASE.cap))
    ma, mb = checks.scored(*a), checks.scored(*b)
    assert checks.effects(ma, mb, ma)["spec_effect"] == 0.0


def test_the_table_reports_a_variation_that_only_loses_months_as_a_sample_effect():
    x = _p(44, start="1975-01-01")
    book = e.Book.single(x, _rf(x), "A")
    per = ex.holding_periods(book)
    res = runs.run(runs.prepare(book, filters.rolling), 1.0, per)
    short = res.loc["2000-01":]                                     # the same run, without its first years
    assets = {"screened": ["mkt"], "fixed": ["mkt"]}
    R = {}
    for fam in ("screened", "fixed"):
        for _, _, change in checks.variations(assets[fam]):
            spec = checks.spec_of({k: v for k, v in change.items() if k != "era"})
            R[(fam, spec)] = (book, short if spec.drop else res, per)
    T, V, _ = build.table(R, assets, build.base_months(R))
    row = T[(T["base"] == "fixed") & (T["variation"] == "drop market")].iloc[0]
    base = T[(T["base"] == "fixed") & T["is_base"] & (T["check"] == "frequency")].iloc[0]
    assert row["months_lost"] > 0 and row["spec_effect"] == 0.0
    assert row["sample_effect"] == pytest.approx(row["gain"] - base["gain"], abs=1e-12)
    assert (T.loc[T["is_base"], ["spec_effect", "sample_effect"]] == 0).all().all()


# ---------------------------------------------------------------- finding 8: a tie is not a flip

def test_a_gain_of_exactly_zero_is_a_tie_not_a_flip():
    assert checks.verdict(0.3, "edge", [0.0, 0.2], ["edge", "edge"]) == "holds"
    assert checks.verdict(0.3, "edge", [0.0, 0.2], ["inconclusive", "edge"]) == "sign holds"
    assert checks.verdict(0.0, "inconclusive", [0.1, -0.1], ["inconclusive"] * 2) == "holds"
    assert checks.verdict(0.3, "edge", [-0.01], ["inconclusive"]) == "flips"


# ---------------------------------------------------------------- finding 9: the mechanics report

def test_the_mechanics_report_names_a_run_with_no_whole_month():
    x = _p()
    book = e.Book.single(x, _rf(x), "A", freq=DAY)
    res = pd.DataFrame({"entry": [True] + [False] * 9}, index=book.positions.index[10:20])   # ten days inside a month
    line = build.run_line("fixed", checks.Spec(freq=DAY), book, res, pd.period_range("1995-01", periods=12, freq=MONTH))
    assert "no whole months" in line and "base months not scored: 12" in line
    assert build.span(pd.PeriodIndex([], freq=MONTH)) == "no whole months"


# ---------------------------------------------------------------- finding 10: the screened family needs survivors

def test_the_screened_family_without_survivors_says_so():
    with pytest.raises(ValueError, match="survivors"):
        checks.recipes(_panel(), "screened")


# ---------------------------------------------------------------- finding 11: the trading calendar, once

def test_a_monthly_decision_never_builds_the_trading_calendar(monkeypatch):
    x = _p()
    book = e.Book.single(x, _rf(x), "A")
    prep = e.prepare(book, filters.rolling)
    calls = []
    to_period = pd.DatetimeIndex.to_period
    monkeypatch.setattr(pd.DatetimeIndex, "to_period", lambda self, *a, **k: calls.append(1) or to_period(self, *a, **k))
    e.decide(book, prep, 1.0, min_months=24)
    assert not calls
    daily = e.Book.single(x, _rf(x), "A", freq=DAY)
    assert daily.calendar is daily.calendar


# ---------------------------------------------------------------- finding 12: crisis paths, once per run

def test_the_crisis_tables_build_each_runs_path_once(monkeypatch):
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("1926-07-01", "2010-12-31", name="date")
    panel = pd.DataFrame({"mkt": rng.normal(0.0003, 0.01, len(idx)), "mom": rng.normal(0.0003, 0.01, len(idx)), "rf": 0.0001}, index=idx)

    def mom_run(*_):
        b = e.Book.single(panel["mom"], panel["rf"], "mom")
        per = ex.holding_periods(b)
        return b, runs.run(runs.prepare(b, filters.rolling), 1.0, per), per
    C = {("mom", "bsc"): {"as run": crises.managed(panel, "mom", "bsc")}, ("mom", "frozen"): {"as run": mom_run()},
         ("mkt", "bsc"): {"as run": crises.managed(panel, "mkt", "bsc"), "cap 1.0": crises.managed(panel, "mkt", "bsc", 1.0)}}
    calls = []
    path = crises.path
    monkeypatch.setattr(crises, "path", lambda *a: calls.append(1) or path(*a))
    Q, _, _ = build.crisis_tables(panel, C)
    assert len(calls) == len(C) and len(Q) > 0


# ---------------------------------------------------------------- finding 14: the placebo, one driver

def test_the_walk_forward_placebo_returns_its_draws_and_the_same_alphas_as_before():
    panel = _panel()
    book = checks.book(panel, checks.recipes(panel, "screened", _survivors(panel, lambda y: ["mkt", "bab"] if y >= 1998 else [])))
    per = ex.holding_periods(book)
    out = wfb.placebo(book, filters.rolling, per, draws=12, jobs=1)
    prep = runs.prepare(book, filters.rolling)
    before = []                                                     # the placebo as Phase 7 computed it
    for s in range(12):
        m = runs.scored(runs.run(prep, 1.0, per, seed=s), per)
        before.append(ev.alpha(m["excess"], m["bench_excess"]))
    assert np.array_equal(out["shuffled"], np.array(before))
    assert out["draws"].shape == (12, len(runs.scored(runs.run(prep, 1.0, per), per)))
    assert np.array_equal(out["draws"].loc[3].to_numpy(), runs.scored(runs.run(prep, 1.0, per, seed=3), per)["excess"].to_numpy())


# ---------------------------------------------------------------- invariants

CHANGES = [{"rule": "inv_vol"}, {"rule": "down_vol"}, {"window": "rolling 10 years"}, {"cap": None}, {"cap": 1.0},
           {"costs": (10e-4, 0.0)}]


@pytest.mark.parametrize("change", CHANGES, ids=str)
def test_timing_one_asset_alone_is_timing_the_mix_under_every_change(cheap, change):
    spec = checks.Spec(**change)
    panel = _panel()
    rec = checks.recipes(panel, "screened", _survivors(panel, lambda y: ["mkt"] if y >= 1999 else []))
    book, comb, per = checks.run(panel, rec, spec)
    each = checks.time_each(book, per, **checks.resolve(spec))
    pd.testing.assert_frame_equal(_accounts(comb), _accounts(each), check_exact=True)


def test_timing_one_asset_alone_is_timing_the_mix(cheap):
    panel = _panel()
    rec = checks.recipes(panel, "screened", _survivors(panel, lambda y: ["mkt"] if y >= 1999 else []))
    _, comb, _ = checks.run(panel, rec, {})
    _, each, _ = checks.run(panel, rec, {"order": "each"})
    pd.testing.assert_frame_equal(_accounts(comb), _accounts(each), check_exact=True)


def _levered_book():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("1990-01-01", periods=252 * 3)
    x = pd.DataFrame({"A": rng.normal(0.0004, 0.01, len(idx)), "B": rng.normal(0.0, 0.01, len(idx))}, index=idx)
    return x, pd.Series(0.0001, idx)


@pytest.mark.parametrize("freq", [MONTH, WEEK, DAY])
def test_the_daily_ledger_books_what_the_engine_books(freq):
    """Monthly: the ledger's months are the engine's monthly path exactly. Weekly and daily: compounded over
    the whole run, the ledger's months give the engine's periods (costs, shorts and borrowing included)."""
    x, rf = _levered_book()
    held = x.index[(x.index >= "1990-02-01") & (x.index <= "1992-11-30")].to_period(freq).unique()
    book = e.Book(x, rf, pd.DataFrame({"A": 1.2, "B": -0.3}, index=held))
    costs = e.Costs(kappa={"A": 1e-3, "B": 2e-3}, borrow_spread=0.005, short_fee={"B": 0.01})
    per = ex.holding_periods(book, costs)
    dec = pd.DataFrame({"size": np.linspace(0.5, 1.8, len(held)), "forecast": 1.0, "capped": False}, index=held)
    res = e.trade(book, dec, per)
    led = ledger.months(book, res, per)
    if freq == MONTH:
        eng = runs.scored(res, per)
        for col in ("excess", "bench_excess"):
            assert np.array_equal(led[col].to_numpy(), eng[col].to_numpy()), col
        return
    days = x.loc[held[0].start_time:held[-1].end_time].index
    cash = np.prod(1 + rf.loc[days])
    for pre in ("", "bench_"):
        total = np.prod(1 + led[f"{pre}excess"] + led["rf"])
        assert total == pytest.approx(np.prod(1 + res[f"{pre}net"]), rel=1e-12)
    assert np.prod(1 + led["rf"]) == pytest.approx(cash, rel=1e-13)
