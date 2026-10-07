"""
Phase 1 build: one trading calendar, excess returns, sealed holdout
===================================================================
    python -m src.data.build              # reuse cached snapshots, fetch what is missing
    python -m src.data.build --refresh    # take a fresh snapshot of every source
    python -m src.data.build --relock     # reseal the holdout after a deliberate data change

Writes data/processed/, data/events/event_dates.csv and
reports/phase1_data_report.md, then exits 0 if every pass-rule check passed.

Output (data/processed/)
------------------------
  panel_dev.parquet, panel_holdout.parquet
      One row per trading day:
        mkt smb hml mom rmw cma bab   factor excess returns (Mkt-RF, long-short)
        spy mtum qual usmv            ETF total return minus rf
        rf                            daily T-bill return (French library)
        vix                           CBOE VIX close (level)
        fomc cpi jobs                 True on the trading day the news first
                                      reaches the market; <NA> before coverage
      Returns are decimals, close to close.
  etf_ohlc_dev.parquet, etf_ohlc_holdout.parquet
      Long table (date, ticker): split-adjusted OHLC and adj_close as Yahoo
      publishes them, plus adj_open/high/low scaled by adj_close/close so that
      ranges and overnight gaps sit on the same total-return basis.

Calendar
--------
The CRSP trading calendar in the French 3-factor file (from 1926-07-01,
including NYSE Saturday sessions until May 1952). It ends on that file's
last day, because rf, and so every ETF excess return, ends there.

  * Returns dated on non-trading days (AQR's global calendar has US-holiday
    rows) are compounded into the next trading day, so nothing is dropped.
  * ETF returns are computed on the calendar: a missing price day gives a
    missing return, never a silent two-day return.
  * Levels (VIX) on non-trading days are dropped. A missing VIX day inside
    its history is filled with the previous close, the value known then.
  * A release on a non-trading day is flagged on the next trading day.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
import pandas as pd

from src.config import (DEV_END, ETFS, EVENTS, EVENTS_CSV, HOLDOUT_LOCK, HOLDOUT_START, PROCESSED_DIR)
from src.data import events
from src.data.download import RawPaths, fetch_all
from src.data.load import content_hash
from src.data.parse import (read_aqr, read_cboe_vix, read_damodaran_sp500, read_french, read_yahoo)


@dataclass
class Build:
    panel: pd.DataFrame
    ohlc: pd.DataFrame
    events: pd.DataFrame
    aux: dict = field(default_factory=dict)   # inputs and notes the checks need


# ---------------------------------------------------------------- alignment

def align_returns(r: pd.Series, cal: pd.DatetimeIndex) -> tuple[pd.Series, pd.DatetimeIndex]:
    """Put returns on the calendar, compounding any off-calendar day into the next
    trading day. Returns the aligned series and the off-calendar dates absorbed."""
    r = r.dropna()
    pos = cal.searchsorted(r.index, side="left")
    inside = pos < len(cal)
    r, target = r[inside], cal[pos[inside]]
    off = r.index[r.index != target]
    out = pd.Series(1.0 + r.to_numpy(), index=target).groupby(level=0).prod() - 1.0
    return out.reindex(cal), off


def total_returns(px: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """Close-to-close total return from adjusted closes, on the calendar."""
    px = px[px.index <= cal[-1]]
    stray = px.index.difference(cal)
    if len(stray):
        raise ValueError(f"price dates off the trading calendar: {list(stray[:5].date)}")
    return px["adj_close"].reindex(cal).pct_change(fill_method=None)


def vix_on_calendar(v: pd.Series, cal: pd.DatetimeIndex):
    """Returns (series on calendar, dropped off-calendar dates, filled gap dates)."""
    v = v[v.index <= cal[-1]]
    dropped = v.index.difference(cal)
    s = v.reindex(cal)
    inside = pd.Series((cal >= v.index[0]) & (cal <= v.index[-1]), index=cal)
    filled = s.index[s.isna() & inside]
    s = s.ffill(limit=1).where(inside)
    filled = filled[s.loc[filled].notna().to_numpy()]
    return s, dropped, filled


def event_flags(table: pd.DataFrame, cal: pd.DatetimeIndex):
    """Returns ({event: boolean flag series}, {event: [(release, flagged day)] shifted})."""
    used = table[table["used"]]
    start = events.coverage_start(table)
    flags, shifted = {}, {}
    for e in EVENTS:
        d = pd.DatetimeIndex(used.loc[used["event"] == e, "date"])
        d = d[d <= cal[-1]]
        day = cal[cal.searchsorted(d, side="left")]
        shifted[e] = [(a, b) for a, b in zip(d, day) if a != b]
        f = pd.Series(False, index=cal, dtype="boolean")
        f.loc[day] = True
        f[cal < start[e]] = pd.NA
        flags[e] = f
    return flags, shifted


def ohlc_table(raw: dict[str, pd.DataFrame], cal: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    for ticker, px in raw.items():
        px = px[px.index <= cal[-1]]
        px = px.reindex(cal[(cal >= px.index[0]) & (cal <= px.index[-1])])
        k = px["adj_close"] / px["close"]
        px = px.assign(adj_open=px["open"] * k, adj_high=px["high"] * k, adj_low=px["low"] * k)
        frames.append(px.assign(ticker=ticker).rename_axis("date").reset_index())
    cols = ["date", "ticker", "open", "high", "low", "close", "adj_open", "adj_high", "adj_low",
            "adj_close", "volume", "dividends", "splits"]
    return pd.concat(frames, ignore_index=True)[cols]


# ---------------------------------------------------------------- build

def build_panel(paths: RawPaths) -> Build:
    ff3, ff5, mom = read_french(paths.ff3), read_french(paths.ff5), read_french(paths.mom)
    cal = ff3.index
    for name, df in (("5-factor", ff5), ("momentum", mom)):
        stray = df.index.difference(cal)
        if len(stray):
            raise ValueError(f"French {name} file has dates off the CRSP calendar: {list(stray[:5])}")
    rf = ff3["RF"]

    panel = pd.DataFrame(index=cal)
    panel["mkt"], panel["smb"], panel["hml"] = ff3["Mkt-RF"], ff3["SMB"], ff3["HML"]
    panel["mom"] = mom["Mom"].reindex(cal)
    panel["rmw"], panel["cma"] = ff5["RMW"].reindex(cal), ff5["CMA"].reindex(cal)
    panel["bab"], bab_off = align_returns(read_aqr(paths.aqr, "BAB Factors"), cal)

    etf_raw = {t: read_yahoo(paths.yahoo[t]) for t in ETFS}
    for t, px in etf_raw.items():
        panel[t.lower()] = total_returns(px, cal) - rf
    panel["rf"] = rf
    panel["vix"], vix_dropped, vix_filled = vix_on_calendar(read_cboe_vix(paths.vix), cal)

    table = events.build_event_table(paths)
    flags, shifted = event_flags(table, cal)
    for e in EVENTS:
        panel[e] = flags[e]

    aux = {
        "etf_raw": etf_raw,
        "aqr_mkt": align_returns(read_aqr(paths.aqr, "MKT"), cal)[0],
        "damodaran": read_damodaran_sp500(paths.damodaran),
        "yahoo_vix": read_yahoo(paths.yahoo["VIX"])["close"],
        "bab_off_calendar": bab_off,
        "vix_dropped": vix_dropped,
        "vix_filled": vix_filled,
        "event_shifts": shifted,
        "event_coverage": events.coverage_start(table),
    }
    return Build(panel.rename_axis("date"), ohlc_table(etf_raw, cal), table, aux)


# ---------------------------------------------------------------- split and seal

def split(b: Build) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    d = b.ohlc["date"]
    dev = {"panel": b.panel.loc[:DEV_END], "etf_ohlc": b.ohlc[d <= DEV_END].reset_index(drop=True)}
    hold = {"panel": b.panel.loc[HOLDOUT_START:],
            "etf_ohlc": b.ohlc[d >= HOLDOUT_START].reset_index(drop=True)}
    return dev, hold


def write_outputs(b: Build, *, seal: bool, relock: bool) -> str:
    """Write panels and the event table. Seals the holdout (writes its hash) when
    seal is True and it is not sealed yet, or relock is True. Refuses to overwrite a
    sealed holdout with different content. Returns a one-line seal status."""
    dev, hold = split(b)
    hashes = {f"{k}_holdout.parquet": content_hash(v) for k, v in hold.items()}
    lock = json.loads(HOLDOUT_LOCK.read_text()) if HOLDOUT_LOCK.exists() else None
    if lock and lock["files"] != hashes and not relock:
        raise RuntimeError("The rebuilt holdout differs from the sealed one (a source revised its "
                           "history, or the build changed). Nothing was written. Re-run with "
                           "--relock only if this change is deliberate.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for k, df in dev.items():
        df.to_parquet(PROCESSED_DIR / f"{k}_dev.parquet")
    for k, df in hold.items():
        path = PROCESSED_DIR / f"{k}_holdout.parquet"
        df.to_parquet(path)
        # The seal must survive a round trip, or load_holdout would reject good data.
        if content_hash(pd.read_parquet(path)) != hashes[path.name]:
            raise RuntimeError(f"{path.name} does not read back to the content it was written with")
    EVENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = b.events.assign(date=b.events["date"].dt.date, file_date=b.events["file_date"].dt.date)
    out.to_csv(EVENTS_CSV, index=False)

    if lock and lock["files"] == hashes:
        return f"holdout unchanged; sealed {lock['sealed_at']}"
    if not seal:
        return "holdout NOT sealed: checks failed"
    HOLDOUT_LOCK.write_text(json.dumps({
        "sealed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dev_end": str(DEV_END.date()), "holdout_start": str(HOLDOUT_START.date()),
        "holdout_last_day": str(b.panel.index[-1].date()),
        "rows": {k: len(v) for k, v in hold.items()},
        "files": hashes,
    }, indent=2) + "\n")
    return "holdout resealed" if lock else "holdout sealed"


def main(argv: list[str] | None = None) -> int:
    from src.data import checks

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--refresh", action="store_true", help="re-download every source")
    ap.add_argument("--relock", action="store_true",
                    help="reseal the holdout after a deliberate change to its data")
    args = ap.parse_args(argv)

    print("snapshot ...")
    paths = fetch_all(refresh=args.refresh)
    print("build ...")
    b = build_panel(paths)
    print("checks ...")
    results = checks.run_all(b)
    passed = all(c.passed for c in results if c.gating)
    seal_status = write_outputs(b, seal=passed, relock=args.relock)
    report = checks.write_report(b, results, seal_status)
    for c in results:
        print(f"  [{'PASS' if c.passed else 'FAIL'}{'' if c.gating else ' info'}] {c.name}: {c.summary}")
    print(f"{'PASS' if passed else 'FAIL'}: {seal_status}. Report: {report}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
