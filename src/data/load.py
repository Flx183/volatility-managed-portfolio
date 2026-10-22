"""
Loading the Phase 1 panels
==========================
    from src.data.load import load_dev, load_etf_ohlc_dev

    panel = load_dev()          # 1926-07-01 .. 2018-12-31, one row per trading day
    ohlc = load_etf_ohlc_dev()  # long table: date, ticker, OHLC, adj_*

The holdout (2019-01-01 to 2026-08-31) was sealed for the Phase 9 final test:

    load_holdout(confirm=HOLDOUT_PASSPHRASE)

checks the data against the hash recorded when it was sealed
(data/holdout_lock.json) and appends a line to data/holdout_access_log.csv,
so the repository shows every time the holdout was opened.

From Phase 10 it is Holdout B: test data only, one run per frozen strategy, through
src/retest with load_holdout(confirm=RETEST_PASSPHRASE) (PREREGISTRATION Entry 15).
Holdout C (2026-09-01 onward) is sealed: no loader returns a row after 2026-08-31.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from src.config import HOLDOUT_ACCESS_LOG, HOLDOUT_END, HOLDOUT_LOCK, PROCESSED_DIR

HOLDOUT_PASSPHRASE = "phase 9 final test"
RETEST_PASSPHRASE = "holdout B retest"        # Phase 10 on: one run of a frozen strategy (src/retest)


def content_hash(df: pd.DataFrame) -> str:
    """Hash of values, index, column names and dtypes (stable across parquet writers)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    h.update(json.dumps([str(t) for t in df.dtypes]).encode())
    return h.hexdigest()


def _read(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; build it with `python -m src.data.build`")
    return pd.read_parquet(path)


def load_dev() -> pd.DataFrame:
    return _read("panel_dev.parquet")


def load_etf_ohlc_dev() -> pd.DataFrame:
    return _read("etf_ohlc_dev.parquet")


def load_holdout(*, confirm: str, table: str = "panel") -> pd.DataFrame:
    """table='panel' or 'etf_ohlc'. Raises unless confirm is HOLDOUT_PASSPHRASE (Phase 9) or RETEST_PASSPHRASE
    (Phase 10 on, through src/retest), and if the table holds a day of Holdout C."""
    if confirm not in (HOLDOUT_PASSPHRASE, RETEST_PASSPHRASE):
        raise PermissionError("The holdout is test data only, opened once per frozen strategy through src/retest. "
                              f"Pass confirm={RETEST_PASSPHRASE!r} to open it; the access is logged.")
    name = f"{table}_holdout.parquet"
    df = _read(name)
    sealed = json.loads(HOLDOUT_LOCK.read_text())["files"][name]
    if content_hash(df) != sealed:
        raise RuntimeError(f"{name} does not match the hash it was sealed with; it has been changed")
    last = df.index.max() if table == "panel" else df["date"].max()
    if last > HOLDOUT_END:
        raise RuntimeError(f"{name} runs to {last.date()}, into Holdout C (sealed after {HOLDOUT_END.date()})")
    new = not HOLDOUT_ACCESS_LOG.exists()
    with HOLDOUT_ACCESS_LOG.open("a") as f:
        if new:
            f.write("opened_at_utc,user,table,command\n")
        cmd = " ".join(sys.argv).replace(",", " ")
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')},{getpass.getuser()},{table},{cmd}\n")
    return df
