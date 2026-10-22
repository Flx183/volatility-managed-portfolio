"""
Phase 1 pass rule
=================
Gating checks (all must pass before Phase 2):

  calendar         one CRSP trading calendar; Yahoo's ETF dates agree with it exactly
  complete         every series has no missing trading day between its first and last day
  spy_annual       SPY annual total returns match an independent source (Damodaran's S&P 500)
  alignment        per source, same-day correlation with the market dominates the +-1 day ones
  known_extremes   1987-10-19 and 2008-10-13 sit where history says; the extremes are listed
  yahoo_internal   adjusted-close returns agree with close + dividend returns
  bounds           no impossible values
  events           one CPI and one jobs release per month; 7-8 scheduled FOMC meetings a year
  split            development and holdout do not overlap and, with the sealed Holdout C rows after
                   2026-08-31 (never written), cover the panel

What looks at the holdout: dates, completeness, and pass/fail counts from the
bounds and Yahoo checks. Nothing that reports return values (annual returns,
correlations, extreme days) reads past 2018.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEV_END, ETFS, EVENTS, FACTORS, HOLDOUT_END, MANIFEST, REPORTS_DIR
from src.data.build import Build, forward_rows, split

ETF_COLS = [t.lower() for t in ETFS]
RETURNS = FACTORS + ETF_COLS

# Known stress episodes, used to label the largest daily moves. A label means the
# day falls inside a documented episode; unlabelled extremes are listed for review.
EPISODES = [
    ("1929-10-01", "1929-12-31", "1929 crash"),
    ("1930-01-01", "1933-12-31", "Great Depression, bank panics"),
    ("1937-08-01", "1938-12-31", "1937-38 recession"),
    ("1939-09-01", "1940-06-30", "outbreak of WWII, fall of France"),
    ("1950-06-26", "1950-06-27", "Korean War outbreak"),
    ("1955-09-26", "1955-09-26", "Eisenhower heart attack"),
    ("1962-05-28", "1962-05-29", "1962 flash crash"),
    ("1970-05-25", "1970-05-28", "1970 bear-market low and rebound"),
    ("1987-10-01", "1987-12-31", "1987 crash"),
    ("1989-10-13", "1989-10-16", "October 1989 mini-crash"),
    ("1997-10-27", "1997-10-28", "Asian crisis"),
    ("1998-08-01", "1998-10-31", "Russia default, LTCM"),
    ("1999-10-01", "2002-12-31", "dot-com boom and bust"),
    ("2007-07-01", "2009-06-30", "global financial crisis (incl. 2007 quant and 2009 momentum crashes)"),
    ("2010-05-06", "2010-05-07", "flash crash"),
    ("2011-08-01", "2011-10-31", "US downgrade, euro crisis"),
    ("2015-08-20", "2015-08-26", "China devaluation sell-off"),
    ("2016-06-24", "2016-06-27", "Brexit vote"),
    ("2018-02-05", "2018-02-08", "Volmageddon"),
    ("2018-10-10", "2018-10-11", "October 2018 tech sell-off"),
    ("2018-12-24", "2018-12-26", "December 2018 sell-off"),
]
# October 2025 CPI and jobs reports were never published (federal funding lapse).
UNPUBLISHED_MONTHS = {pd.Period("2025-10", "M")}


@dataclass
class Check:
    name: str
    passed: bool
    summary: str
    details: str = ""
    gating: bool = True


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _d(ts) -> str:
    return "" if ts is None or pd.isna(ts) else str(pd.Timestamp(ts).date())


def _episode(ts: pd.Timestamp) -> str:
    for a, b, label in EPISODES:
        if pd.Timestamp(a) <= ts <= pd.Timestamp(b):
            return label
    return "**review**"


# ---------------------------------------------------------------- checks

def check_calendar(b: Build) -> Check:
    cal = b.panel.index
    ordered = cal.is_monotonic_increasing and cal.is_unique
    sundays = int((cal.dayofweek == 6).sum())
    late_saturdays = int(((cal.dayofweek == 5) & (cal > "1952-06-01")).sum())
    rows, disagree = [], 0
    for t, px in b.aux["etf_raw"].items():
        d = px.index[px.index <= cal[-1]]
        extra, missing = d.difference(cal), cal[cal >= d[0]].difference(d)
        disagree += len(extra) + len(missing)
        rows.append([t, _d(d[0]), f"{len(d):,}", len(extra), len(missing)])
    gaps = pd.Series((cal[1:] - cal[:-1]).days, index=cal[1:])
    closures = [[_d(cal[cal.get_loc(i) - 1]), _d(i), n] for i, n in gaps.nlargest(6).items()]
    passed = ordered and sundays == 0 and late_saturdays == 0 and disagree == 0
    summary = (f"{len(cal):,} CRSP trading days {_d(cal[0])} to {_d(cal[-1])}; "
               + ("ETF dates from Yahoo agree exactly" if disagree == 0 else f"{disagree} ETF dates disagree"))
    details = ("Saturday sessions end 1952-05-24 (NYSE stopped Saturday trading); "
               f"Sundays: {sundays}; Saturdays after 1952: {late_saturdays}.\n\n"
               + _table(["ETF", "first day", "days", "off calendar", "calendar days missing"], rows)
               + "\n\nLongest market closures (should be known events, e.g. the March 1933 bank "
                 "holiday and September 2001):\n\n"
               + _table(["last open day", "reopened", "calendar days"], closures))
    return Check("calendar", passed, summary, details)


def check_complete(b: Build) -> Check:
    rows, missing_total = [], 0
    for col in RETURNS + ["rf", "vix"]:
        s = b.panel[col]
        first, last = s.first_valid_index(), s.last_valid_index()
        inside = s.loc[first:last]
        missing = int(inside.isna().sum())
        missing_total += missing
        rows.append([col, _d(first), _d(last), f"{len(inside):,}", missing])
    for e in EVENTS:
        inside = b.panel[e].loc[b.aux["event_coverage"][e]:]
        missing = int(inside.isna().sum())
        missing_total += missing
        rows.append([e, _d(inside.index[0]), _d(inside.index[-1]), f"{len(inside):,}", missing])
    filled = b.aux["vix_filled"]
    summary = (f"no missing trading days in {len(rows)} series" if missing_total == 0
               else f"{missing_total} missing trading days")
    details = (_table(["series", "first", "last", "trading days", "missing"], rows)
               + f"\n\nVIX: {len(filled)} single-day gaps in the CBOE file filled with the previous "
                 f"close ({', '.join(_d(d) for d in filled) or 'none'}).")
    return Check("complete", missing_total == 0, summary, details)


def check_spy_annual(b: Build) -> Check:
    dev = b.panel.loc[:DEV_END]
    total = (dev["spy"] + dev["rf"]).dropna()
    first_full = total.index[0].year + 1                 # listing year is partial
    total = total[total.index.year >= first_full]
    spy = total.groupby(total.index.year).apply(lambda r: (1 + r).prod() - 1)
    ref = b.aux["damodaran"].reindex(spy.index)
    diff = spy - ref
    passed = bool(ref.notna().all() and diff.abs().max() <= 0.01)
    summary = (f"{len(spy)} years {spy.index[0]}-{spy.index[-1]}: max |SPY - S&P 500| "
               f"{diff.abs().max():.2%}, mean {diff.mean():+.2%}")
    rows = [[y, f"{spy[y]:+.2%}", f"{ref[y]:+.2%}", f"{diff[y]:+.2%}"] for y in spy.index]
    details = ("Independent source: Damodaran (NYU Stern), S&P 500 annual return including "
               "dividends. SPY should trail by its expense ratio (0.09-0.20% a year) plus "
               "dividend cash drag, and Damodaran adds dividends without reinvesting them "
               "within the year. Tolerance: 1 percentage point in every year.\n\n"
               + _table(["year", "SPY (Yahoo, rebuilt as spy + rf)", "S&P 500 (Damodaran)", "difference"], rows))
    return Check("spy_annual", passed, summary, details)


def check_alignment(b: Build) -> Check:
    dev = b.panel.loc[:DEV_END]
    m = dev["mkt"]
    series = {"SPY": dev["spy"], "MTUM": dev["mtum"], "QUAL": dev["qual"], "USMV": dev["usmv"],
              "AQR MKT, USA (same file and dates as BAB)": b.aux["aqr_mkt"].loc[:DEV_END],
              "VIX daily change": dev["vix"].diff()}
    rows, passed = [], True
    for name, x in series.items():
        df = pd.concat({"x": x, "m": m, "m_prev": m.shift(1), "m_next": m.shift(-1)}, axis=1).dropna()
        c0, cp, cn = (df["x"].corr(df[k]) for k in ("m", "m_prev", "m_next"))
        ok = abs(c0) >= 0.5 and abs(c0) >= 3 * max(abs(cp), abs(cn))
        passed &= ok
        rows.append([name, f"{len(df):,}", f"{c0:+.3f}", f"{cp:+.3f}", f"{cn:+.3f}", "ok" if ok else "FAIL"])
    summary = "same-day correlation with Mkt-RF dominates for every source" if passed else "a source looks shifted"
    details = ("If a source's dates were shifted by a day, its correlation with the market would "
               "move from lag 0 to lag +-1. Rule: |corr at 0| >= 0.5 and >= 3x both neighbours. "
               "Development sample only.\n\n"
               + _table(["series vs Mkt-RF", "days", "same day", "vs day before", "vs day after", ""], rows))
    return Check("alignment", passed, summary, details)


def check_known_extremes(b: Build) -> Check:
    dev = b.panel.loc[:DEV_END]
    mkt, spy, vix = dev["mkt"], dev["spy"], dev["vix"]
    t87, t08, tvix = pd.Timestamp("1987-10-19"), pd.Timestamp("2008-10-13"), pd.Timestamp("2008-11-20")
    tests = [
        ("Mkt-RF 1987-10-19 is the worst day, below -15%", mkt.idxmin() == t87 and mkt[t87] < -0.15, f"{mkt[t87]:+.2%}"),
        ("Mkt-RF 2008-10-13 above +10%", mkt[t08] > 0.10, f"{mkt[t08]:+.2%}"),
        ("SPY 2008-10-13 above +10%", spy[t08] > 0.10, f"{spy[t08]:+.2%}"),
        ("highest VIX close is 2008-11-20", vix.idxmax() == tvix, f"{vix.max():.2f} on {_d(vix.idxmax())}"),
    ]
    passed = all(ok for _, ok, _ in tests)
    rows, review = [], 0
    for col in RETURNS:
        s = dev[col].dropna()
        z = (s - s.mean()) / s.std()
        for ts in s.abs().nlargest(5).index:
            label = _episode(ts)
            review += label == "**review**"
            rows.append([col, _d(ts), f"{s[ts]:+.2%}", f"{z[ts]:+.1f}", label])
    summary = (f"1987-10-19 Mkt-RF {mkt[t87]:+.2%}, 2008-10-13 {mkt[t08]:+.2%}; "
               f"{review} of {len(rows)} top-5 extremes outside a listed episode; nothing trimmed")
    details = (_table(["test", "result", "value"], [[n, "ok" if ok else "FAIL", v] for n, ok, v in tests])
               + "\n\nFive largest absolute daily moves per series (development sample). Extreme "
                 "days are the signal: they are confirmed against history, never trimmed.\n\n"
               + _table(["series", "date", "return", "z", "episode"], rows))
    return Check("known_extremes", passed, summary, details)


def check_yahoo_internal(b: Build) -> Check:
    end = b.panel.index[-1]
    rows, worst = [], 0.0
    for t, px in b.aux["etf_raw"].items():
        px = px[px.index <= end]
        r_adj = px["adj_close"].pct_change()
        r_div = (px["close"] + px["dividends"]) / px["close"].shift() - 1
        gap = float((r_adj - r_div).abs().max())
        worst = max(worst, gap)
        rows.append([t, int((px["dividends"] > 0).sum()), int((px["splits"] > 0).sum()), f"{gap:.1e}"])
    passed = worst <= 5e-4
    summary = f"adjusted-close and close+dividend returns agree within {worst:.1e} per day"
    details = ("Two ways to get a total return from Yahoo: the adjusted close, and the close plus "
               "the cash dividend on its ex-date. They should differ only by Yahoo's rounding "
               "(tolerance 5e-4 per day).\n\n"
               + _table(["ETF", "dividends", "splits", "max daily difference"], rows))
    return Check("yahoo_internal", passed, summary, details)


def check_bounds(b: Build) -> Check:
    p = b.panel
    tests = {
        "non-finite returns": int(np.isinf(p[RETURNS + ["rf"]].to_numpy()).sum()),
        "ETF total return <= -100%": int(((p[ETF_COLS].add(p["rf"], axis=0)) <= -1).sum().sum()),
        "factor |return| >= 50% in a day": int((p[FACTORS].abs() >= 0.5).sum().sum()),
        "rf outside [-0.1%, 1%] a day": int(((p["rf"] < -0.001) | (p["rf"] > 0.01)).sum()),
        "VIX outside (0, 150)": int(((p["vix"] <= 0) | (p["vix"] >= 150)).sum()),
    }
    bad = sum(tests.values())
    return Check("bounds", bad == 0, "no impossible values" if bad == 0 else f"{bad} impossible values",
                 _table(["test (whole panel, counts only)", "violations"], [[k, v] for k, v in tests.items()]))


def check_events(b: Build) -> Check:
    used = b.events[b.events["used"]]
    cal = b.panel.index
    rows, problems = [], []
    for e in ("cpi", "jobs"):
        u = used[used["event"] == e]
        refs = pd.PeriodIndex(u["label"].str[:7], freq="M")
        expected = pd.period_range(refs.min(), refs.max(), freq="M")
        gaps = sorted(set(expected) - set(refs))
        unexpected = [str(g) for g in gaps if g not in UNPUBLISHED_MONTHS]
        weekend = int((u["date"].dt.dayofweek >= 5).sum())
        parsed = (u["date_source"] == "embargo line").mean()
        if unexpected:
            problems.append(f"{e}: no release for {unexpected}")
        if weekend:
            problems.append(f"{e}: {weekend} releases dated on a weekend")
        if u["date"].duplicated().any():
            problems.append(f"{e}: two releases on one day")
        if parsed < 0.9:
            problems.append(f"{e}: embargo date read from only {parsed:.0%} of releases")
        if u["date"].max() < cal[-1] - pd.Timedelta(days=45):
            problems.append(f"{e}: archive looks stale (last release {_d(u['date'].max())})")
        rows.append([e, len(u), f"{refs.min()} to {refs.max()}", ", ".join(map(str, gaps)) or "none",
                     f"{parsed:.1%}", _d(u["date"].min())])

    f = used[used["event"] == "fomc"]
    per_year = f.groupby(f["date"].dt.year).size()
    per_year = per_year.loc[:cal[-1].year - 1]          # complete years only
    off = per_year[per_year != 8]
    if (~per_year.isin([7, 8])).any():
        problems.append(f"fomc: unusual meeting counts {per_year[~per_year.isin([7, 8])].to_dict()}")
    rows.append(["fomc", len(f), f"{per_year.index.min()} to {f['date'].max().year}",
                 "years with != 8: " + (", ".join(f"{y} ({n})" for y, n in off.items()) or "none"),
                 "n/a", "1994-01-01"])

    for e in EVENTS:
        start = b.aux["event_coverage"][e]
        n_events = used.loc[(used["event"] == e) & (used["date"] >= start) & (used["date"] <= cal[-1]), "date"].nunique()
        n_flags = int(b.panel[e].sum())
        if n_events != n_flags:
            problems.append(f"{e}: {n_events} events but {n_flags} flagged days")

    mismatch = used[(used["event"] != "fomc") & (used["date_source"] == "embargo line")
                    & (used["date"] != used["file_date"])]
    shifts = [[e, _d(a), _d(d)] for e, s in b.aux["event_shifts"].items() for a, d in s]
    excluded = b.events[(b.events["event"] == "fomc") & ~b.events["used"]]
    passed = not problems
    summary = (f"{len(used):,} scheduled releases/meetings; {len(shifts)} moved to the next trading day"
               if passed else "; ".join(problems))
    details = (_table(["event", "count", "span", "months / years missing", "date from embargo line", "flags from"], rows)
               + "\n\nA missing reference month is allowed only when BLS never published it "
                 "(October 2025, federal funding lapse).\n\n"
               + f"Releases whose archive file name gives a different date than the embargo line "
                 f"({len(mismatch)}; the embargo line is used):\n\n"
               + _table(["event", "label", "file name date", "embargo date"],
                        [[r.event, r.label, _d(r.file_date), _d(r.date)] for r in mismatch.itertuples()])
               + "\n\nReleases on a non-trading day, flagged on the next trading day:\n\n"
               + _table(["event", "released", "flagged"], shifts)
               + f"\n\nFOMC items not flagged because they were not on the advance calendar: {len(excluded)} "
                 "(unscheduled meetings, conference calls, notation votes, cancelled meetings), e.g. "
               + ", ".join(excluded["label"].head(6)) + ".")
    return Check("events", passed, summary, details)


def check_split(b: Build) -> Check:
    dev, hold = split(b)
    fwd = forward_rows(b)
    p_dev, p_hold = dev["panel"], hold["panel"]
    first_2019 = b.panel.index[b.panel.index.year == 2019][0]
    passed = (p_dev.index.max() <= DEV_END and p_hold.index[0] == first_2019 and p_hold.index[-1] <= HOLDOUT_END
              and len(p_dev) + len(p_hold) + fwd["panel"] == len(b.panel)
              and len(dev["etf_ohlc"]) + len(hold["etf_ohlc"]) + fwd["etf_ohlc"] == len(b.ohlc)
              and dev["etf_ohlc"]["date"].max() <= DEV_END)
    summary = (f"development {_d(p_dev.index[0])} to {_d(p_dev.index[-1])} ({len(p_dev):,} days); "
               f"holdout {_d(p_hold.index[0])} to {_d(p_hold.index[-1])} ({len(p_hold):,} days), separate files")
    if fwd["panel"]:
        summary += f"; {fwd['panel']:,} days after {_d(HOLDOUT_END)} sealed as Holdout C, not written"
    return Check("split", passed, summary)


def check_vix_sources(b: Build) -> Check:
    dev = b.panel["vix"].loc[:DEV_END].dropna()
    y = b.aux["yahoo_vix"].reindex(dev.index).dropna()
    diff = (dev.loc[y.index] - y).abs()
    share = float((diff <= 0.05).mean())
    summary = (f"CBOE and Yahoo VIX closes agree within 0.05 on {share:.1%} of {len(y):,} development days; "
               f"{len(b.aux['vix_dropped'])} CBOE rows on non-trading days dropped")
    worst = diff.nlargest(5)
    details = _table(["date", "CBOE", "Yahoo"], [[_d(d), f"{dev[d]:.2f}", f"{y[d]:.2f}"] for d in worst.index])
    return Check("vix_sources", share >= 0.99, summary, details, gating=False)


def check_aqr_calendar(b: Build) -> Check:
    off = b.aux["bab_off_calendar"]
    summary = (f"{len(off)} AQR rows on US non-trading days ({_d(off.min())} to {_d(off.max())}) "
               "compounded into the next trading day" if len(off) else "AQR dates all on the calendar")
    details = ("AQR builds its daily file on a global calendar. On a US holiday the US BAB "
               "factor still moves slightly (its financing leg accrues, -rf x (1/beta_L - 1/beta_H)), "
               "so those rows are compounded into the next US trading day rather than dropped.")
    return Check("aqr_calendar", True, summary, details, gating=False)


def run_all(b: Build) -> list[Check]:
    return [f(b) for f in (check_calendar, check_complete, check_spy_annual, check_alignment,
                           check_known_extremes, check_yahoo_internal, check_bounds, check_events,
                           check_split, check_vix_sources, check_aqr_calendar)]


# ---------------------------------------------------------------- report

def write_report(b: Build, results: list[Check], seal_status: str) -> Path:
    passed = all(c.passed for c in results if c.gating)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    sources = [[f"`{k}`", v["url"], v["fetched_at"][:10], f"{v['bytes']:,}", v["sha256"][:12]]
               for k, v in sorted(manifest.items()) if not k.startswith("fomc/fomchistorical")]
    n_hist = sum(k.startswith("fomc/fomchistorical") for k in manifest)
    lines = [
        "# Phase 1 data report",
        "",
        f"Generated by `python -m src.data.build` on "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        f"**Pass rule: {'PASS' if passed else 'FAIL'}.** {seal_status.capitalize()}.",
        "",
        _table(["check", "result", "summary"],
               [[c.name, ("PASS" if c.passed else "FAIL") + ("" if c.gating else " (info)"), c.summary]
                for c in results]),
        "",
        "Only dates, completeness and violation counts are reported for the holdout (2019 onward). "
        "Every figure built from return values uses the development sample.",
        "",
    ]
    for c in results:
        if c.details:
            lines += [f"## {c.name}", "", c.details, ""]
    lines += ["## sources", "",
              _table(["raw file", "url", "fetched", "bytes", "sha256"], sources),
              "",
              f"Plus {n_hist} yearly FOMC history pages (fomchistoricalYYYY.htm) and the BLS "
              "release documents read for embargo dates; full hashes in `data/raw_manifest.json`.",
              ""]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "phase1_data_report.md"
    path.write_text("\n".join(lines))
    return path
