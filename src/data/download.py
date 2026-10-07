"""
Raw data snapshots
==================
Downloads every Phase 1 source into data/raw/ and records each file's URL,
fetch time, size and SHA-256 in data/raw_manifest.json.

Files already on disk are reused, not re-downloaded, unless refresh=True.
The French library and AQR both rebuild their full history on every update,
so the snapshot, not the live URL, is what makes a build reproducible. A raw
file whose hash no longer matches the manifest stops the build.

Sources
-------
  french/     Fama-French 3 factors, 5 factors (2x3) and momentum, daily
  aqr/        Betting Against Beta equity factors, daily (BAB and MKT sheets)
  yahoo/      SPY MTUM QUAL USMV via yfinance; ^VIX for a cross-check only
  cboe/       VIX daily history. FRED's VIXCLS republishes this same CBOE
              close; CBOE is used directly because FRED's CSV endpoint is
              unreliable for scripted downloads.
  damodaran/  S&P 500 annual total returns (independent spot check for SPY)
  fomc/       Federal Reserve meeting calendars, 1994 onward
  bls/        CPI and Employment Situation archive pages, plus each release
              document (read for its embargo date; cached, not in manifest)
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

from src.config import MANIFEST, RAW_DIR, ETFS
from src.data import events

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "ff3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom": "F-F_Momentum_Factor_daily_CSV.zip",
}
AQR_BAB_URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
               "Betting-Against-Beta-Equity-Factors-Daily.xlsx")
CBOE_VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
DAMODARAN_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xlsx"
FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FOMC_HISTORICAL_URL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
BLS_ARCHIVE_URL = events.BLS_BASE + "/bls/news-release/{series}.htm"
BLS_SERIES = {"cpi": "cpi", "jobs": "empsit"}

# bls.gov rejects clients that do not look like a browser, and its filter
# fingerprints the HTTP stack: curl over HTTP/2 passes where requests fails.
_BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
    "sec-ch-ua": '"Chromium";v="126"',
    "sec-ch-ua-platform": '"macOS"',
}


@dataclass
class RawPaths:
    ff3: Path
    ff5: Path
    mom: Path
    aqr: Path
    vix: Path
    damodaran: Path
    fomc_calendar: Path
    yahoo: dict[str, Path] = field(default_factory=dict)        # ticker -> csv
    fomc_historical: dict[int, Path] = field(default_factory=dict)
    bls_archive: dict[str, Path] = field(default_factory=dict)  # event -> archive page
    bls_docs: dict[str, Path] = field(default_factory=dict)     # event -> release dir


# ---------------------------------------------------------------- HTTP

def http_get(url: str, *, timeout: float = 120, retries: int = 4) -> bytes:
    err = ""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": _BROWSER_HEADERS["User-Agent"]},
                             timeout=timeout)
            if r.status_code == 200:
                return r.content
            err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            err = str(e)
        if attempt < retries - 1:
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"GET {url} failed after {retries} tries: {err}")


def curl_get(url: str, *, timeout: float = 60, retries: int = 5) -> bytes:
    cmd = ["curl", "-sS", "-L", "--compressed", "--fail", "--max-time", str(timeout)]
    for k, v in _BROWSER_HEADERS.items():
        cmd += ["-A", v] if k == "User-Agent" else ["-H", f"{k}: {v}"]
    err = ""
    for attempt in range(retries):
        out = subprocess.run(cmd + [url], capture_output=True)
        if out.returncode == 0 and out.stdout:
            return out.stdout
        err = out.stderr.decode(errors="replace").strip() or "empty response body"
        if attempt < retries - 1:
            time.sleep(10 * 2 ** attempt)
    raise RuntimeError(f"curl {url} failed after {retries} tries: {err}")


def yahoo_csv(ticker: str) -> bytes:
    import yfinance as yf

    df = yf.Ticker(ticker).history(period="max", auto_adjust=False, actions=True)
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    df.index = df.index.tz_localize(None).normalize()
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                            "Adj Close": "adj_close", "Volume": "volume",
                            "Dividends": "dividends", "Stock Splits": "splits"})
    cols = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]
    buf = io.StringIO()
    df[cols].rename_axis("date").to_csv(buf, float_format="%.10g")
    return buf.getvalue().encode()


# ---------------------------------------------------------------- snapshot

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Snapshot:
    """Fetch-once file store under RAW_DIR with a hash manifest."""

    def __init__(self, refresh: bool = False):
        self.refresh = refresh
        self.manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}

    def get(self, rel: str, url: str, getter: Callable[[str], bytes], *,
            record: bool = True) -> Path:
        path = RAW_DIR / rel
        # Unrecorded files are immutable documents (BLS releases): never re-fetched.
        if path.exists() and path.stat().st_size and (not self.refresh or not record):
            if record:
                digest = _sha256(path.read_bytes())
                entry = self.manifest.get(rel)
                if entry is None:
                    self._record(rel, url, path.read_bytes(), fetched="unknown (pre-existing file)")
                elif entry["sha256"] != digest:
                    raise RuntimeError(f"{rel} differs from the manifest hash; "
                                       "re-run with --refresh to take a new snapshot")
            return path
        data = getter(url)
        if not data:
            raise RuntimeError(f"{url} returned an empty body")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if record:
            self._record(rel, url, data)
        print(f"  fetched {rel} ({len(data):,} bytes)")
        return path

    def _record(self, rel: str, url: str, data: bytes, fetched: str | None = None) -> None:
        self.manifest[rel] = {
            "url": url,
            "fetched_at": fetched or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(data),
            "sha256": _sha256(data),
        }

    def save(self) -> None:
        MANIFEST.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------- sources

def fetch_all(refresh: bool = False) -> RawPaths:
    snap = Snapshot(refresh)
    try:
        paths = RawPaths(
            **{k: snap.get(f"french/{f}", FRENCH_BASE + f, http_get) for k, f in FRENCH_FILES.items()},
            aqr=snap.get("aqr/bab_daily.xlsx", AQR_BAB_URL, lambda u: http_get(u, timeout=900)),
            vix=snap.get("cboe/VIX_History.csv", CBOE_VIX_URL, http_get),
            damodaran=snap.get("damodaran/histretSP.xlsx", DAMODARAN_URL, http_get),
            fomc_calendar=snap.get("fomc/fomccalendars.htm", FOMC_CALENDAR_URL, http_get),
        )
        for t in ETFS + ["^VIX"]:
            name = t.lstrip("^")
            paths.yahoo[name] = snap.get(f"yahoo/{name}.csv", f"yfinance:{t}", lambda u: yahoo_csv(u.split(":")[1]))

        # The live calendar page covers the most recent years; older years each have a page.
        cal_years = events.fomc_calendar_years(paths.fomc_calendar.read_text(errors="replace"))
        for y in range(events.FOMC_FIRST_YEAR, min(cal_years)):
            paths.fomc_historical[y] = snap.get(f"fomc/fomchistorical{y}.htm",
                                                FOMC_HISTORICAL_URL.format(year=y), http_get)

        for event, series in BLS_SERIES.items():
            page = snap.get(f"bls/{series}_archive.htm", BLS_ARCHIVE_URL.format(series=series), curl_get)
            paths.bls_archive[event] = page
            paths.bls_docs[event] = RAW_DIR / "bls" / series
            listing = events.parse_bls_archive(page.read_text(errors="replace"), series)
            for doc in listing["doc_path"]:
                if doc.endswith(".pdf"):
                    continue                    # no text to read; filename date is used
                rel = f"bls/{series}/{Path(doc).name}"
                marker = RAW_DIR / (rel + ".unavailable")
                if marker.exists():
                    continue
                if not (RAW_DIR / rel).exists():
                    time.sleep(0.3)             # be polite to bls.gov
                try:
                    snap.get(rel, events.BLS_BASE + doc, curl_get, record=False)
                except RuntimeError as e:
                    # e.g. cpi_06162016.htm: bls.gov serves an empty page (the PDF is fine).
                    # The release keeps its file-name date; the marker stops re-tries.
                    marker.write_text(f"{datetime.now(timezone.utc).isoformat()} {e}\n")
                    print(f"  unavailable: {rel} (file-name date used)")
    finally:
        snap.save()
    return paths
