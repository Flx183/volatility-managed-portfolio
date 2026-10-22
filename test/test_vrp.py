"""Track C, Phase C0: the VRP, the predictive tests, the gate and the PutWrite data, on simulated data only."""

import json

import numpy as np
import pandas as pd
import pytest

from src.engine.data import MONTH
from src.frozen import Frozen
from src.vrp import TRACK_C, eda, putwrite
from src.walkforward import stats


def _panel(start="1989-01-02", end="1992-12-31", vix=20.0):
    idx = pd.bdate_range(start, end, name="date")
    return pd.DataFrame({"mkt": 0.0004, "rf": 0.0001, "vix": vix}, index=idx)


def _months(n, start="1990-01"):
    return pd.period_range(start, periods=n, freq=MONTH)


def test_vrp_is_the_implied_monthly_variance_less_the_forecast_both_at_the_month_end_close():
    panel = _panel()
    panel.loc["1990-01-31", "vix"] = 30.0                          # the last trading day of January sets it
    months = pd.period_range("1989-01", "1992-12", freq=MONTH)
    V = eda.vrp(panel, {"M0": pd.Series(0.002, months)})
    assert V.index[0] == pd.Period("1990-01", MONTH) and V.index[-1] == pd.Period("1992-12", MONTH)
    assert V.loc["1990-01", "implied"] == pytest.approx(0.09 / 12)
    assert V.loc["1990-02", "vrp_M0"] == pytest.approx(0.04 / 12 - 0.002)
    with pytest.raises(ValueError, match="lacks"):
        eda.vrp(panel, {"M0": pd.Series(0.002, months[:20])})


def test_the_return_ahead_compounds_months_t_plus_one_to_t_plus_h():
    m = _months(5)
    mon = pd.DataFrame({"ret": [0.01, 0.02, -0.03, 0.04, 0.05], "rf": 0.001}, index=m)
    r3 = eda.ahead(mon, 3)
    total = np.prod(1 + mon["ret"].iloc[1:4] + 0.001) - (1.001) ** 3
    assert r3.iloc[0] == pytest.approx(total)
    assert r3.iloc[2:].isna().all() and eda.ahead(mon, 1).iloc[0] == pytest.approx(0.02)
    assert [eda.lags(h) for h in TRACK_C["horizons"]] == [6, 8, 11]


def _predictor(n=348, b=0.5, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    m = _months(n)
    y = pd.Series(np.r_[b * x[:-1] + rng.normal(0, 1, n - 1), np.nan], m)      # y_t is the return of t+1
    return pd.Series(x, m), y


def test_a_real_predictor_passes_every_condition_and_noise_does_not_reach_the_level():
    x, y = _predictor()
    r = eda.predictive(x, y, 1)
    assert r["months"] == 347 and r["p"] < 1e-6 and r["eras_hold"] and r["r2_oos"] > 0.1
    assert r["oos_first"] == "1995-01"                             # at its close, pairs 1990-01 .. 1994-12 are complete
    _, noise = _predictor(b=0.0, seed=1)
    assert eda.predictive(x, noise, 1)["p"] > 0.05


def test_the_out_of_sample_prediction_at_t_reads_no_return_that_ends_after_t():
    x, y = _predictor(n=200)
    y3 = y.rolling(3).sum().shift(-2)                              # an overlapping 3-month target
    P = eda.oos_predictions(x, y3, 3)
    t = pd.Period("1998-06", MONTH)
    later = y3.copy()
    later.loc[t - 2:] += 50                                        # pairs s > t - 3 end after t's close
    P2 = eda.oos_predictions(x, later, 3)
    assert P.loc[:t, ["pred", "bench"]].equals(P2.loc[:t, ["pred", "bench"]])
    assert not P.loc[t + 1:, "pred"].equals(P2.loc[t + 1:, "pred"])


def test_the_eras_are_equal_thirds_and_holm_runs_across_all_nine_cells():
    x, y = _predictor()
    r = eda.predictive(x, y, 1)
    assert (r["era1"], r["era2"], r["era3"]) == ("1990-01 to 1999-08", "1999-09 to 2009-04", "2009-05 to 2018-11")
    V = pd.DataFrame({f"vrp_{m}": x for m in TRACK_C["models"]})
    T = eda.tests(V, {h: y for h in TRACK_C["horizons"]})
    assert len(T) == 9
    np.testing.assert_allclose(T["p_holm"], stats.holm(T["p"].to_numpy()))


def test_the_gate_needs_all_three_conditions_in_the_same_cell():
    T = pd.DataFrame({"model": ["M0", "M4", "M6"], "h": [1, 3, 6], "slope": [1.0, 1.0, -1.0],
                      "p_holm": [0.01, 0.01, 0.001], "eras_hold": [False, True, True], "r2_oos": [0.02, -0.01, 0.05]})
    G, passed = eda.gate(T)
    assert not passed and G["sign_holm"].tolist() == [True, True, False]   # a negative slope never counts
    T.loc[1, "r2_oos"] = 0.001
    assert eda.gate(T)[1] and eda.gate(T)[0]["passes"].tolist() == [False, True, False]


def test_putwrite_returns_start_at_the_first_full_month_and_net_out_the_rf():
    panel = _panel("2006-01-02", "2007-06-29")
    days = panel.index
    put = pd.DataFrame({"put": 100 * 1.01 ** np.arange(len(days))}, index=days)
    put = put.drop(days[(days.month == 2) & (days.year == 2006)][:3])                # February 2006 is not whole
    put = put.loc["2006-02-01":]
    out = eda.monthly_put(put, panel)
    assert out.index[0] == pd.Period("2006-04", MONTH)             # March is the first whole month; April its first return
    lv = put["put"]
    r = lv.loc["2006-04"].iloc[-1] / lv.loc["2006-03"].iloc[-1] - 1
    rf = (1.0001) ** len(days[(days.year == 2006) & (days.month == 4)]) - 1
    assert out.loc["2006-04", "ret"] == pytest.approx(r - rf) and out.loc["2006-04", "rf"] == pytest.approx(rf)
    with pytest.raises(ValueError, match="misses a trading day"):
        eda.monthly_put(put.drop(pd.Timestamp("2006-05-15")), panel)


def test_putwrite_writes_development_and_seals_holdout_b_and_never_writes_holdout_c(tmp_path, monkeypatch):
    days = pd.bdate_range("2018-12-24", "2026-09-10")
    raw = tmp_path / "PUT_History.csv"
    pd.DataFrame({"DATE": days.strftime("%m/%d/%Y"), "PUT": 100.0}).to_csv(raw, index=False)
    monkeypatch.setattr(putwrite, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(putwrite, "DEV_FILE", tmp_path / "dev.parquet")
    monkeypatch.setattr(putwrite, "HOLDOUT_FILE", tmp_path / "hold.parquet")
    lock = tmp_path / "lock.json"
    assert putwrite.prepare(raw, lock) == "putwrite: sealed"
    dev, hold = pd.read_parquet(tmp_path / "dev.parquet"), pd.read_parquet(tmp_path / "hold.parquet")
    assert dev.index.max() == pd.Timestamp("2018-12-31") and hold.index.max() == pd.Timestamp("2026-08-31")
    assert json.loads(lock.read_text())["rows"] == {"dev": len(dev), "holdout": len(hold)}
    assert putwrite.prepare(raw, lock) == "putwrite: unchanged, sealed"
    pd.DataFrame({"DATE": days.strftime("%m/%d/%Y"), "PUT": 101.0}).to_csv(raw, index=False)
    with pytest.raises(RuntimeError, match="differ from the sealed"):
        putwrite.prepare(raw, lock)


def test_the_rules_are_read_only():
    assert isinstance(TRACK_C, Frozen) and TRACK_C["models"] == ["M0", "M4", "M6"]
    with pytest.raises(TypeError, match="frozen"):
        TRACK_C["oos_min_pairs"] = 36
