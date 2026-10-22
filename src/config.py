"""
Project-wide paths and the development / holdout split.

Every phase imports its dates from here, so the split is defined exactly once.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                       # downloaded snapshots (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"           # built panels (git-ignored)
EVENTS_CSV = DATA_DIR / "events" / "event_dates.csv"
MANIFEST = DATA_DIR / "raw_manifest.json"        # URL, time, size, SHA-256 per raw file
HOLDOUT_LOCK = DATA_DIR / "holdout_lock.json"    # content hash of the sealed holdout
HOLDOUT_ACCESS_LOG = DATA_DIR / "holdout_access_log.csv"
REPORTS_DIR = ROOT / "reports"

# Development: start of data .. DEV_END. All EDA, choices and tuning happen here.
# Holdout (Holdout B from Phase 10): HOLDOUT_START .. HOLDOUT_END. Opened in Phase 9; from Phase 10 it is test
#   data only, one run per frozen strategy (PREREGISTRATION Entry 15).
# Holdout C: FORWARD_START onward, sealed. No strategy runs on it until frozen; nothing writes it to a file yet.
DEV_END = pd.Timestamp("2018-12-31")
HOLDOUT_START = pd.Timestamp("2019-01-01")
HOLDOUT_END = pd.Timestamp("2026-08-31")
FORWARD_START = pd.Timestamp("2026-09-01")

FACTORS = ["mkt", "smb", "hml", "mom", "rmw", "cma", "bab"]
# ETFs start in 2011 (USMV) to 2013 (MTUM, QUAL): 5-7 years before DEV_END, under the 10-year initial
# training window. They are never tuned on; every choice comes from the paper factors.
ETFS = ["SPY", "MTUM", "QUAL", "USMV"]
EVENTS = ["fomc", "cpi", "jobs"]
