import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data import build, load, parse

CAL = pd.DatetimeIndex(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-05", "2024-07-08"])


def test_off_calendar_returns_are_compounded_not_dropped():
    # 4 July is a US holiday: AQR still has a row for it.
    r = pd.Series([0.01, -0.0003, 0.02], index=pd.to_datetime(["2024-07-03", "2024-07-04", "2024-07-05"]))
    out, off = build.align_returns(r, CAL)
    assert list(off) == [pd.Timestamp("2024-07-04")]
    assert out["2024-07-03"] == pytest.approx(0.01)
    assert out["2024-07-05"] == pytest.approx((1 - 0.0003) * 1.02 - 1)
    assert np.isnan(out["2024-07-01"])
    assert (1 + out.dropna()).prod() == pytest.approx((1 + r).prod())


def test_missing_price_day_never_becomes_a_two_day_return():
    px = pd.DataFrame({"adj_close": [100.0, 101.0, 103.0, 104.0]},
                      index=pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-05", "2024-07-08"]))
    r = build.total_returns(px, CAL)
    assert np.isnan(r["2024-07-03"]) and np.isnan(r["2024-07-05"])   # gap shows up as missing
    assert r["2024-07-08"] == pytest.approx(104 / 103 - 1)


def test_price_on_a_non_trading_day_is_an_error():
    px = pd.DataFrame({"adj_close": [100.0, 101.0]}, index=pd.to_datetime(["2024-07-03", "2024-07-04"]))
    with pytest.raises(ValueError, match="off the trading calendar"):
        build.total_returns(px, CAL)


def test_vix_single_gap_filled_with_previous_close_and_holiday_dropped():
    v = pd.Series([12.0, 13.0, 14.0, 15.0],
                  index=pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-04", "2024-07-05"]))
    s, dropped, filled = build.vix_on_calendar(v, CAL)
    assert list(dropped) == [pd.Timestamp("2024-07-04")]
    assert list(filled) == [pd.Timestamp("2024-07-03")]
    assert s["2024-07-03"] == 13.0 and s["2024-07-05"] == 15.0
    assert np.isnan(s["2024-07-08"])          # after the last close: not filled


def test_vix_two_day_hole_is_left_for_the_completeness_check():
    v = pd.Series([12.0, 13.0, 16.0], index=pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-08"]))
    s, _, filled = build.vix_on_calendar(v, CAL)
    assert list(filled) == [pd.Timestamp("2024-07-03")]
    assert np.isnan(s["2024-07-05"])


def test_event_flags_shift_to_next_trading_day_and_are_na_before_coverage():
    table = pd.DataFrame({
        "event": ["jobs", "jobs", "cpi", "fomc"],
        "date": pd.to_datetime(["2024-07-02", "2024-07-06", "2024-07-03", "2024-07-03"]),
        "used": [True, True, True, False],
    })
    flags, shifted = build.event_flags(table, CAL)
    assert flags["jobs"].tolist()[1:] == [True, False, False, True]     # Saturday -> Monday
    assert pd.isna(flags["jobs"]["2024-07-01"])                       # before first release
    assert shifted["jobs"] == [(pd.Timestamp("2024-07-06"), pd.Timestamp("2024-07-08"))]
    assert not flags["fomc"].any()                                    # unscheduled: never flagged


def test_read_french_handles_preamble_missing_codes_and_percent(tmp_path):
    text = ("This file was created using the CRSP database.\r\n\r\n"
            ",Mkt-RF,SMB,RF\r\n"
            "19871016,  -5.00,  0.10, 0.02\r\n"
            "19871019, -17.44,-99.99, 0.03\r\n"
            "\r\nCopyright 2026 Eugene F. Fama and Kenneth R. French\r\n")
    path = tmp_path / "ff.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ff.csv", text)
    df = parse.read_french(path)
    assert list(df.columns) == ["Mkt-RF", "SMB", "RF"]
    assert df.loc["1987-10-19", "Mkt-RF"] == pytest.approx(-0.1744)
    assert np.isnan(df.loc["1987-10-19", "SMB"])
    assert len(df) == 2


def test_holdout_is_sealed(tmp_path, monkeypatch):
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2019-01-02", "2019-01-03"]))
    df.to_parquet(tmp_path / "panel_holdout.parquet")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"files": {"panel_holdout.parquet": load.content_hash(df)}}))
    log = tmp_path / "log.csv"
    monkeypatch.setattr(load, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(load, "HOLDOUT_LOCK", lock)
    monkeypatch.setattr(load, "HOLDOUT_ACCESS_LOG", log)

    with pytest.raises(PermissionError):
        load.load_holdout(confirm="just peeking")
    assert not log.exists()

    out = load.load_holdout(confirm=load.HOLDOUT_PASSPHRASE)
    pd.testing.assert_frame_equal(out, df)
    assert len(log.read_text().splitlines()) == 2                    # header + one access

    df.assign(x=[1.0, 2.5]).to_parquet(tmp_path / "panel_holdout.parquet")
    with pytest.raises(RuntimeError, match="does not match"):
        load.load_holdout(confirm=load.HOLDOUT_PASSPHRASE)
