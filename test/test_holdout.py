"""Phase 9: the joined panel, where each book's data end, the crisis spans, Lo's standard error and the reading
table, on simulated data only. The whole build is rehearsed on development data with
`python -m src.holdout.build --rehearse`."""

import numpy as np
import pandas as pd
import pytest

from src.engine import evaluation as ev
from src.holdout import DEVELOPMENT, EPISODES, HOLDOUT_OUTCOME, HOLDOUT_RULES, VERDICT, setup
from src.walkforward import PRIMARY, RULES


def _panel(start="2017-01-02", end="2020-12-31", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, end, name="date")
    return pd.DataFrame({"mkt": rng.normal(0.0004, 0.01, len(idx)), "bab": rng.normal(0.0003, 0.005, len(idx)),
                         "rf": 0.0001}, index=idx)


def test_the_panel_joins_development_then_holdout_and_refuses_an_overlap_or_other_columns():
    p = _panel()
    dev, hold = p.loc[:"2018-12-31"], p.loc["2019-01-01":]
    assert setup.join(dev, hold).equals(p)
    with pytest.raises(ValueError, match="starts on"):
        setup.join(dev, p.loc["2018-12-31":])
    with pytest.raises(ValueError, match="columns"):
        setup.join(dev, hold[["bab", "mkt", "rf"]])
    o = pd.DataFrame({"date": p.index, "ticker": "SPY", "close": 1.0})
    j = setup.join(o.iloc[:10], o.iloc[10:], date="date")
    assert j.equals(o)


def test_the_paper_books_end_with_the_last_month_every_factor_is_complete():
    p = _panel()
    p.loc["2020-12-01":, "bab"] = np.nan                    # AQR's file ends a month early
    assert setup.complete_end(p, ["mkt", "bab"]) == pd.Timestamp("2020-11-30")
    assert setup.complete_end(p, ["mkt"]) == pd.Timestamp("2020-12-31")
    p.loc["2020-11-20", "bab"] = np.nan                     # a gap inside the month: that month is not complete
    assert setup.complete_end(p, ["mkt", "bab"]) == pd.Timestamp("2020-10-30")
    paper, ohlc = setup.cut(p, pd.DataFrame({"date": p.index, "x": 1.0}), pd.Timestamp("2020-10-30"))
    assert paper.index[-1] == ohlc["date"].iloc[-1] == pd.Timestamp("2020-10-30")


def test_only_decembers_are_year_ends_and_the_datas_last_day_is_not_one():
    idx = pd.bdate_range("2017-01-02", "2020-07-31")
    assert list(setup.december_ends(idx)) == [pd.Timestamp(d) for d in ("2017-12-29", "2018-12-31", "2019-12-31")]


def test_a_crisis_trough_is_the_years_lowest_close_and_its_peak_the_highest_before_it():
    """2020's case: the market makes a new high by December, so Entry 8's rule (the peak is the highest close
    of the first year) would put the peak after the crash. The trough comes first, then the peak before it."""
    idx = pd.bdate_range("2019-06-03", "2021-12-31")
    anchors = {"2019-06-03": 1.0, "2019-11-01": 1.30,       # higher than the 2020 peak, but before peak_from
               "2019-12-31": 1.0, "2020-02-19": 1.20,       # the peak
               "2020-03-23": 0.80,                          # the trough
               "2020-12-30": 1.40, "2021-12-31": 1.50}      # a new high by the year's end
    level = pd.Series(np.nan, idx)
    for d, x in anchors.items():
        level.loc[d] = x
    level = level.interpolate(method="time")
    r = level.pct_change().fillna(0.0)
    panel = pd.DataFrame({"mkt": r, "rf": 0.0}, index=idx)
    s = setup.span(panel, *EPISODES["2020"])
    assert (s["peak"], s["trough"]) == (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"))
    assert s["start"] == pd.Timestamp("2019-08-19") and s["end"] == pd.Timestamp("2021-03-23")
    assert s["market_fall"] == pytest.approx(0.80 / 1.20 - 1)
    assert s["months"][0] == pd.Period("2019-08", "M") and s["months"][-1] == pd.Period("2021-03", "M")
    s22 = setup.span(panel, "2019-06-03", "2020")            # a search from earlier finds the earlier high
    assert s22["peak"] == pd.Timestamp("2019-11-01")


def test_lo_reproduces_the_plans_worked_example():
    ex = setup.lo_example(0.5, 92)
    assert ex["monthly_sharpe"] == pytest.approx(0.144, abs=5e-4)
    assert ex["se_monthly"] == pytest.approx(0.105, abs=5e-4)
    assert ex["se_annual"] == pytest.approx(0.36, abs=5e-3)
    assert ex["half_width_95"] == pytest.approx(0.71, abs=5e-3)
    x = pd.Series(np.random.default_rng(0).normal(0.01, 0.04, 92))
    sr = x.mean() / x.std()
    assert setup.lo_se(x) == pytest.approx(np.sqrt((1 + sr ** 2 / 2) / 92) * np.sqrt(12))


def test_every_reading_maps_to_a_column_of_the_plans_table_and_every_cell_has_a_verdict():
    readings = {ev.reading({"p": 0.01, "upper": 1.0}), ev.reading({"p": 0.5, "upper": 0.05}), ev.reading({"p": 0.5, "upper": 0.5})}
    assert set(HOLDOUT_OUTCOME) == readings
    for dev in ("wins", "fails"):
        for outcome in HOLDOUT_OUTCOME.values():
            assert (dev, outcome) in VERDICT
    assert {v for (dev, _), v in VERDICT.items() if dev == "fails"} == {"consistent with Cederburg et al.; report it"}
    assert set(DEVELOPMENT) == set(HOLDOUT_RULES["decides"])
    assert all(d in ("wins", "fails") for d, _ in DEVELOPMENT.values())


def test_the_holdout_runs_the_frozen_primary_and_phase_7s_best_variant():
    assert HOLDOUT_RULES["primary"] == PRIMARY == ("M3", "inv_var", "inv_vol", "monthly")
    best = pd.read_csv("reports/phase7_summary.csv", index_col=0)["value"]["best_variant"]
    assert tuple(best.split(" | ")) == HOLDOUT_RULES["best"]
    assert HOLDOUT_RULES["best"][1] in RULES
    assert HOLDOUT_RULES["decides"]["primary"] == "(c)" and HOLDOUT_RULES["decides"]["spy"] == "(a)"


def test_the_final_test_refuses_to_run_again_before_opening_the_holdout(monkeypatch):
    from src.holdout import build
    monkeypatch.setattr(build, "load_holdout", lambda **kw: pytest.fail("the holdout was opened"))
    assert (build.REPORTS_DIR / "phase9_results.csv").exists()
    with pytest.raises(SystemExit, match="already run"):
        build.main(rehearse=False)
