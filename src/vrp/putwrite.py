"""
Track C, data: the CBOE PutWrite index (PUT)
============================================
A new source, kept apart from Phase 1's panel so the sealed holdout's files and hashes are untouched.

    fetch(refresh)     the raw file into data/raw/cboe_put/ (reused if present)
    prepare()          the raw file split as Phase 1 splits the panel: development rows (to 31 December 2018) to
                       data/processed/putwrite_dev.parquet; Holdout B rows (1 January 2019 to 31 August 2026) to
                       putwrite_holdout.parquet, sealed by content hash in data/putwrite_lock.json with the raw
                       file's URL, fetch time and SHA-256; Holdout C rows (after 31 August 2026) not written.
                       Once sealed, a raw file or split that differs is refused
    load_dev()         the development rows (daily index levels)

CBOE's free file holds daily values from January 2007, plus 7 stray rows before. The index's history back to 1986
is not in it, so question 5 reads January 2007 to December 2018. No loader for the Holdout B rows exists yet: a
frozen Track C strategy would open them through src/retest (PREREGISTRATION Entry 15).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import requests

from src.config import DATA_DIR, DEV_END, HOLDOUT_END, HOLDOUT_START, PROCESSED_DIR, RAW_DIR
from src.data.load import content_hash
from src.vrp import PUTWRITE

RAW = RAW_DIR / PUTWRITE["raw"]
LOCK = DATA_DIR / PUTWRITE["lock"]
DEV_FILE, HOLDOUT_FILE = PROCESSED_DIR / "putwrite_dev.parquet", PROCESSED_DIR / "putwrite_holdout.parquet"


def fetch(refresh: bool = False) -> None:
    if RAW.exists() and not refresh:
        return
    r = requests.get(PUTWRITE["url"], timeout=60)
    r.raise_for_status()
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(r.content)


def read_raw(path=RAW) -> pd.DataFrame:
    d = pd.read_csv(path)
    out = pd.DataFrame({"put": d["PUT"].astype(float).to_numpy()},
                       index=pd.DatetimeIndex(pd.to_datetime(d["DATE"], format="%m/%d/%Y"), name="date")).sort_index()
    if out.index.duplicated().any():
        raise ValueError(f"{path.name}: a date appears twice")
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Development and Holdout B. Holdout C's rows go into neither."""
    return df.loc[:DEV_END], df.loc[HOLDOUT_START:HOLDOUT_END]


def prepare(path=RAW, lock=LOCK) -> str:
    raw_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    dev, hold = split(read_raw(path))
    sealed = {"putwrite_holdout.parquet": content_hash(hold)}
    if lock.exists():
        old = json.loads(lock.read_text())
        if old["raw_sha256"] != raw_sha or old["files"] != sealed:
            raise RuntimeError(f"{path.name} or its holdout rows differ from the sealed ones; nothing was written")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(DEV_FILE)
    hold.to_parquet(HOLDOUT_FILE)
    if content_hash(pd.read_parquet(HOLDOUT_FILE)) != sealed["putwrite_holdout.parquet"]:
        raise RuntimeError("putwrite_holdout.parquet does not read back to the content it was written with")
    if lock.exists():
        return "putwrite: unchanged, sealed"
    fetched = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    lock.write_text(json.dumps({
        "url": PUTWRITE["url"], "fetched_at": fetched, "raw_sha256": raw_sha, "bytes": path.stat().st_size,
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dev_last_day": str(dev.index[-1].date()), "holdout_last_day": str(hold.index[-1].date()),
        "rows": {"dev": len(dev), "holdout": len(hold)}, "files": sealed,
    }, indent=2) + "\n")
    return "putwrite: sealed"


def load_dev() -> pd.DataFrame:
    if not DEV_FILE.exists():
        raise FileNotFoundError(f"{DEV_FILE} not found; build it with src.vrp.putwrite.prepare()")
    df = pd.read_parquet(DEV_FILE)
    if df.index.max() > DEV_END:
        raise RuntimeError("putwrite_dev.parquet runs past the development sample")
    return df
