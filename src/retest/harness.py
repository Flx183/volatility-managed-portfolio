"""
Phase 10 on: the Holdout B runner
=================================
Phase 9's runner refuses a second run by design and is left as it is. A Phase 10+ strategy runs on
Holdout B through this one, once, with its code frozen first (PREREGISTRATION Entry 15):

    frozen_at(commit, entry)          the code that runs is the code frozen with the strategy's entry: `commit` is
                                      HEAD or an ancestor of it, its PREREGISTRATION.md holds `entry`, and src/ in
                                      the working tree equals src/ at `commit` (no change since, committed or not,
                                      and no new file)
    start(name, commit, entry, ...)   the checks, in order, before anything is opened: the strategy has no results
                                      yet, then frozen_at. Then the panel: Phase 9's (src/holdout/build.panels),
                                      development + Holdout B through the logged loader (RETEST_PASSPHRASE), the
                                      paper books cut at July 2026 and the ETF track at August 2026
    graded(df, grades)                every row with its variant's grade (GRADES)
    write(run, tables)                reports/phase10_<name>_<table>.csv; the results table is what marks the run done

`rehearse=True` runs the same code on development data only (Phase 9's rehearsal: a pseudo-holdout from 2011)
and writes to data/processed/phase10_rehearsal/<name>/. It needs no frozen code and never opens the holdout.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR, ROOT
from src.data.load import RETEST_PASSPHRASE
from src.holdout import build as hb
from src.retest import GRADES

RESULTS = "results"                      # the table whose file marks a strategy's one run on Holdout B


class NotFrozen(RuntimeError):
    """The code about to run is not the code frozen with the strategy's PREREGISTRATION entry."""


@dataclass
class Run:
    name: str
    rehearse: bool
    out: Path                            # where the tables go
    data: dict                           # Phase 9's panels(): panel, paper, ohlc, etf_panel, start, ends, episodes
    state: dict = field(default_factory=dict)


def results_file(name: str) -> Path:
    return REPORTS_DIR / f"phase10_{name}_{RESULTS}.csv"


def frozen_at(commit: str, entry: str, root: Path = ROOT) -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    found = git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
    if found.returncode:
        raise NotFrozen(f"{commit!r} is not a commit")
    sha = found.stdout.strip()
    if git("merge-base", "--is-ancestor", sha, "HEAD").returncode:
        raise NotFrozen(f"{sha[:12]} is not HEAD or an ancestor of it")
    if entry not in git("show", f"{sha}:PREREGISTRATION.md").stdout:
        raise NotFrozen(f"PREREGISTRATION.md at {sha[:12]} does not hold {entry!r}")
    changed = git("diff", "--name-only", sha, "--", "src").stdout.split()
    new = git("ls-files", "--others", "--exclude-standard", "--", "src").stdout.split()
    if changed or new:
        raise NotFrozen(f"src/ differs from {sha[:12]}: {', '.join(changed + new)}")
    return {"commit": sha, "entry": entry}


def start(name: str, commit: str | None, entry: str, rehearse: bool = False) -> Run:
    if rehearse:
        out = PROCESSED_DIR / "phase10_rehearsal" / name
        state = {"mode": "rehearsal (development data only)"}
    else:
        if results_file(name).exists():
            raise SystemExit(f"{name} has already run on Holdout B ({results_file(name).name} exists); one run per frozen strategy")
        if commit is None:
            raise NotFrozen("a run on Holdout B needs the commit its code was frozen at")
        out = REPORTS_DIR
        state = {"mode": "Holdout B", **frozen_at(commit, entry),
                 "opened_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.mkdir(parents=True, exist_ok=True)
    data = hb.panels(rehearse, confirm=RETEST_PASSPHRASE)
    state |= {"panel_last": data["panel"].index[-1].date(), "first": data["start"].date(),
              "paper_end": data["paper_end"].date(), "etf_end": data["etf_end"].date()}
    return Run(name, rehearse, out, data, state)


def graded(df: pd.DataFrame, grades: dict, key: str = "variant") -> pd.DataFrame:
    g = df[key].map(grades)
    if g.isna().any() or not g.isin(list(GRADES)).all():
        raise ValueError(f"every {key} needs a grade, one of {list(GRADES)}")
    return df.assign(grade=g)


def write(run: Run, tables: dict[str, pd.DataFrame]) -> None:
    """Every table as phase10_<name>_<table>.csv, the run's state as ..._summary.csv, and the results table
    last: its file is what refuses a second run."""
    for key, df in tables.items():
        if key != RESULTS:
            df.to_csv(run.out / f"phase10_{run.name}_{key}.csv", index=False)
    pd.Series(run.state).to_csv(run.out / f"phase10_{run.name}_summary.csv", header=["value"])
    if RESULTS in tables:
        tables[RESULTS].to_csv(run.out / f"phase10_{run.name}_{RESULTS}.csv", index=False)
