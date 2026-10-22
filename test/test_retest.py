"""Phase 10 on: the Holdout C seal, the Holdout B loader, and the runner's run-once and frozen-code checks, on
simulated data and a throwaway git repository. Nothing here opens the real holdout."""

import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.config import FORWARD_START, HOLDOUT_END, HOLDOUT_START
from src.data import build as db, load
from src.retest import CONTAMINATION, GRADES, RETEST_RULES, harness


def _sealed(tmp_path, monkeypatch, df):
    df.to_parquet(tmp_path / "panel_holdout.parquet")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"files": {"panel_holdout.parquet": load.content_hash(df)}}))
    monkeypatch.setattr(load, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(load, "HOLDOUT_LOCK", lock)
    monkeypatch.setattr(load, "HOLDOUT_ACCESS_LOG", tmp_path / "log.csv")


def test_holdout_b_opens_with_the_retest_passphrase_and_every_opening_is_logged(tmp_path, monkeypatch):
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2019-01-02", "2026-08-31"]))
    _sealed(tmp_path, monkeypatch, df)
    with pytest.raises(PermissionError, match="retest"):
        load.load_holdout(confirm="just peeking")
    pd.testing.assert_frame_equal(load.load_holdout(confirm=load.RETEST_PASSPHRASE), df)
    load.load_holdout(confirm=load.HOLDOUT_PASSPHRASE)
    assert len((tmp_path / "log.csv").read_text().splitlines()) == 3        # header + two openings


def test_no_loader_returns_a_day_of_holdout_c(tmp_path, monkeypatch):
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2026-08-31", "2026-09-01"]))
    _sealed(tmp_path, monkeypatch, df)
    with pytest.raises(RuntimeError, match="Holdout C"):
        load.load_holdout(confirm=load.RETEST_PASSPHRASE)
    assert not (tmp_path / "log.csv").exists()


def test_the_data_build_writes_holdout_c_into_no_file():
    idx = pd.bdate_range("2018-12-24", "2026-09-11", name="date")
    panel = pd.DataFrame({"mkt": 0.0}, index=idx)
    ohlc = pd.DataFrame({"date": idx, "ticker": "SPY", "close": 1.0})
    b = db.Build(panel, ohlc, pd.DataFrame())
    dev, hold = db.split(b)
    fwd = db.forward_rows(b)
    assert hold["panel"].index[0] >= HOLDOUT_START and hold["panel"].index[-1] == HOLDOUT_END
    assert hold["etf_ohlc"]["date"].max() == HOLDOUT_END
    assert fwd["panel"] == len(idx[idx >= FORWARD_START]) == fwd["etf_ohlc"] == 9
    assert len(dev["panel"]) + len(hold["panel"]) + fwd["panel"] == len(panel)


def test_the_rules_hold_the_configs_dates_and_every_row_of_the_contamination_log_names_a_bias():
    assert RETEST_RULES["holdout_b"] == (str(HOLDOUT_START.date()), str(HOLDOUT_END.date()))
    assert RETEST_RULES["holdout_c"].startswith(str(FORWARD_START.date()))
    assert set(GRADES) == {"clean", "exploratory"}
    assert all(row["seen"] and row["could_bias"] for row in CONTAMINATION)


@pytest.fixture
def repo(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
                              cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git("init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "PREREGISTRATION.md").write_text("## Entry 16 — 2026-10-21: Track A\n")
    git("add", ".")
    git("commit", "-q", "-m", "freeze")
    return tmp_path, git, git("rev-parse", "HEAD")


def test_the_code_must_be_the_code_frozen_with_the_entry(repo):
    root, git, sha = repo
    assert harness.frozen_at(sha, "## Entry 16", root)["commit"] == sha
    for commit, entry, why in (("nope", "## Entry 16", "not a commit"), (sha, "## Entry 17", "does not hold")):
        with pytest.raises(harness.NotFrozen, match=why):
            harness.frozen_at(commit, entry, root)
    (root / "README.md").write_text("notes\n")                    # a later commit outside src/ is fine
    git("add", "README.md")
    git("commit", "-q", "-m", "notes")
    harness.frozen_at(sha, "## Entry 16", root)
    (root / "src" / "a.py").write_text("x = 2\n")                 # an uncommitted change in src/
    with pytest.raises(harness.NotFrozen, match="a.py"):
        harness.frozen_at(sha, "## Entry 16", root)
    git("commit", "-q", "-am", "change")                           # ... and committed
    with pytest.raises(harness.NotFrozen, match="a.py"):
        harness.frozen_at(sha, "## Entry 16", root)
    git("revert", "--no-edit", "HEAD")
    (root / "src" / "b.py").write_text("y = 1\n")                 # a new file in src/
    with pytest.raises(harness.NotFrozen, match="b.py"):
        harness.frozen_at(sha, "## Entry 16", root)


def test_a_second_run_or_unfrozen_code_is_refused_before_the_holdout_is_opened(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(harness.hb, "panels", lambda *a, **k: pytest.fail("the holdout was opened"))
    with pytest.raises(harness.NotFrozen, match="needs the commit"):
        harness.start("track_a", None, "## Entry 16")
    (tmp_path / "phase10_track_a_results.csv").write_text("done\n")
    with pytest.raises(SystemExit, match="already run"):
        harness.start("track_a", "HEAD", "## Entry 16")


def test_every_row_carries_its_variants_grade_and_the_results_file_is_written_last(tmp_path):
    df = pd.DataFrame({"variant": ["A1", "A2"], "gain": [0.1, np.nan]})
    out = harness.graded(df, {"A1": "exploratory", "A2": "clean"})
    assert out["grade"].tolist() == ["exploratory", "clean"]
    with pytest.raises(ValueError, match="needs a grade"):
        harness.graded(df, {"A1": "exploratory"})
    with pytest.raises(ValueError, match="needs a grade"):
        harness.graded(df, {"A1": "exploratory", "A2": "promising"})
    run = harness.Run("x", False, tmp_path, {}, {"mode": "Holdout B"})
    harness.write(run, {"results": out, "years": df})
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["phase10_x_results.csv", "phase10_x_summary.csv", "phase10_x_years.csv"]
