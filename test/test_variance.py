import numpy as np
import pandas as pd
import pytest

from src import variance as v


def test_har_target_is_the_next_22_days_only():
    v2 = pd.Series(np.arange(100, dtype=float))
    h = v.har_frame(v2)
    assert h.loc[30, "y"] == pytest.approx(v2.loc[31:52].mean())      # t+1 .. t+22, never t itself
    assert h.loc[30, "w"] == pytest.approx(v2.loc[26:30].mean())      # t-4 .. t
    assert h["y"].iloc[-22:].isna().all()                             # no target past the sample end


def test_downside_rv_extends_back_when_fewer_than_three_down_days():
    days = pd.bdate_range("2000-01-03", "2000-02-29")
    jan, feb = days[days.month == 1], days[days.month == 2]
    r = pd.Series(0.01, index=days)
    r[jan[::2]] = -0.01                                              # January: plenty of down days
    r[feb[:2]] = -0.02                                               # February: only two
    m = v.monthly_table(r)
    assert m.loc["2000-02", "extended"] and not m.loc["2000-01", "extended"]
    assert m.loc["2000-02", "rv_down"] == pytest.approx(2 * 0.02 ** 2 + len(jan[::2]) * 0.01 ** 2)
    assert m.loc["2000-02", "rv"] == pytest.approx((r[feb] ** 2).sum())


def test_spanning_t_stat_does_not_depend_on_c():
    rng = np.random.default_rng(0)
    rv = pd.Series(rng.lognormal(-6, 0.8, 300))
    mon = pd.DataFrame({"rv": rv, "ret": rng.normal(0.005, np.sqrt(rv.shift(1).fillna(rv.mean())))})
    a = v.t4_spanning(mon)
    b = v.t4_spanning(mon.assign(rv=mon["rv"] * 7.0))               # rescaling RV rescales c
    assert a["t_alpha"] == pytest.approx(b["t_alpha"])
    assert a["alpha"] == pytest.approx(b["alpha"])                  # c re-normalises the level too
