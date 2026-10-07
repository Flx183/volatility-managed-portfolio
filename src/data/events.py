"""
Scheduled-event calendars
=========================
Days on which scheduled macro news reaches the market.

  fomc   Last day of each *scheduled* FOMC meeting (statement ~2:00 pm ET).
         Unscheduled meetings, conference calls, notation votes and cancelled
         meetings are kept in the table with used=False and never flagged:
         they were not on the calendar in advance, so flagging them would
         leak future information. Coverage starts in 1994, the first year the
         Committee announced its decisions on the day.
  cpi    Consumer Price Index release (8:30 am ET, before the open).
  jobs   Employment Situation release (8:30 am ET, before the open).

BLS release dates come from the embargo line inside each archived release
("...embargoed until 8:30 A.M. (EST), Friday, January 7, 2000"). The date in
the archive's file name is only a fallback, because it is not always the
release date: the December 1999 jobs report is filed as empsit_01192000.txt
but was released on Friday 7 January 2000.
"""

from __future__ import annotations

import calendar
import html as htmllib
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.data.download import RawPaths

FOMC_FIRST_YEAR = 1994
STATEMENT_EVERY_MEETING = pd.Timestamp("1999-05-18")
BLS_BASE = "https://www.bls.gov"

_MONTHS = {m[:3].lower(): i for i, m in enumerate(calendar.month_name) if m}
_FULL_MONTHS = "|".join(calendar.month_name[1:])
_WEEKDAY_DATE = re.compile(
    rf"(?:(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+)?({_FULL_MONTHS})\.?"
    r"\s+(\d{1,2}),?\s*(\d{4})", re.I)


def _text(fragment: str) -> str:
    fragment = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", fragment)
    return " ".join(htmllib.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


# ---------------------------------------------------------------- FOMC

def meeting_end(label: str, year: int) -> pd.Timestamp:
    """'January 31-February 1', 'Jan/Feb 31-1', 'August 5' -> the meeting's last day."""
    months = [_MONTHS[w[:3].lower()] for w in re.findall(r"[A-Za-z]+", label)
              if w[:3].lower() in _MONTHS]
    days = re.findall(r"\d+", label)
    if not months or not days:
        raise ValueError(f"cannot parse FOMC meeting date from {label!r}")
    return pd.Timestamp(year, months[-1], int(days[-1]))


def _is_scheduled(label: str) -> bool:
    # "(unscheduled)", "(cancelled)", "(notation vote)" and "Conference Call" all fail this.
    return "(" not in label and "conference call" not in label.lower()


def parse_fomc_historical(page: str) -> pd.DataFrame:
    """One year's fomchistoricalYYYY.htm: a panel per meeting, headed like
    'January 27-28 Meeting - 2015'. From May 1999 every policy meeting ends with a
    statement, so a 'Meeting' panel without one (e.g. 15 September 2003, agenda only)
    is not a policy meeting. Before that, statements came only with policy changes."""
    parts = re.split(r"<h5[^>]*>(.*?)</h5>", page, flags=re.S)   # [pre, head, body, head, body, ...]
    rows = []
    for raw, body in zip(parts[1::2], parts[2::2]):
        m = re.fullmatch(r"(.+?) - (\d{4})", _text(raw))
        if not m:
            continue
        label, year = m.group(1), int(m.group(2))
        date = meeting_end(label, year)
        has_statement = re.search(r">\s*Statement\s*<", body) is not None
        used = (_is_scheduled(label) and label.endswith("Meeting")
                and (has_statement or date < STATEMENT_EVERY_MEETING))
        rows.append({"date": date, "label": f"{label} - {year}", "used": used})
    return pd.DataFrame(rows, columns=["date", "label", "used"])


def fomc_calendar_years(page: str) -> list[int]:
    return sorted(int(y) for y in re.findall(r">\s*(\d{4}) FOMC Meetings\s*<", page))


def parse_fomc_calendar(page: str) -> pd.DataFrame:
    """fomccalendars.htm: one panel per year, one row per meeting (month + day range)."""
    parts = re.split(r">\s*(\d{4}) FOMC Meetings\s*<", page)
    rows = []
    for year, seg in zip(parts[1::2], parts[2::2]):
        for chunk in seg.split("fomc-meeting__month")[1:]:
            month = re.search(r"<strong>([^<]+)</strong>", chunk)
            days = re.search(r'fomc-meeting__date[^>]*>([^<]+)<', chunk)
            if not (month and days):
                continue
            label = f"{_text(month.group(1))} {_text(days.group(1))}"
            rows.append({"date": meeting_end(label.replace("*", ""), int(year)),
                         "label": f"{label} - {year}", "used": _is_scheduled(label)})
    return pd.DataFrame(rows, columns=["date", "label", "used"])


# ---------------------------------------------------------------- BLS

def _file_date(digits: str) -> pd.Timestamp:
    return pd.to_datetime(digits, format="%m%d%Y" if len(digits) == 8 else "%m%d%y")


def parse_bls_archive(page: str, series: str) -> pd.DataFrame:
    """Archive index (bls.gov/bls/news-release/<series>.htm): one row per reference month."""
    page = re.sub(r"<!--.*?-->", "", page, flags=re.S)   # unreleased placeholders live in comments
    link = re.compile(rf'href="(?:https?://www\.bls\.gov)?'
                      rf'(/news\.release/(?:archives|history)/{series}_(\d{{6}}|\d{{8}})\.(htm|txt|pdf))"')
    prefer = {"htm": 0, "txt": 1, "pdf": 2}
    rows = []
    for li in re.findall(r"<li[^>]*>(.*?)</li>", page, flags=re.S):
        found = link.findall(li)
        ref = re.match(r"([A-Z][a-z]+ \d{4})", _text(li))
        if not (found and ref):
            continue                                       # e.g. "October 2025 - Not published"
        doc, digits, _ = min(found, key=lambda f: prefer[f[2]])
        rows.append({"ref_month": pd.Period(pd.to_datetime(ref.group(1), format="%B %Y"), "M"),
                     "file_date": _file_date(digits), "doc_path": doc})
    df = pd.DataFrame(rows, columns=["ref_month", "file_date", "doc_path"])
    dup = df["ref_month"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{series}: several releases for {sorted(df.loc[dup, 'ref_month'].astype(str))}")
    return df.sort_values("file_date").reset_index(drop=True)


def embargo_date(doc: str) -> pd.Timestamp | None:
    """Release date from the release's header ('... embargoed until 8:30 a.m. (EST)
    [Tuesday,] January 14, 1997', or 'For release: ... Thursday, November 5, 1998').
    A corrected reissue ('Replaces USDL 02-669, issued December 6, 2002') returns the
    original issue date, when the market got the news. None if no date is found, or
    if a printed weekday disagrees with the date: the January 1997 CPI release says
    'Tuesday, January 14, 1996'."""
    text = _text(doc)
    reissue = re.search(rf"Replaces USDL[^,]*, issued ({_FULL_MONTHS})\s+(\d{{1,2}}),?\s*(\d{{4}})", text, re.I)
    if reissue:
        return pd.to_datetime(" ".join(reissue.groups()), format="%B %d %Y")
    anchors = [i for i in (text.lower().find("embargo"), text.lower().find("for release")) if i >= 0]
    if not anchors:
        return None
    i = min(anchors)
    m = _WEEKDAY_DATE.search(text, i, i + 1500)
    if m is None:
        return None
    date = pd.to_datetime(f"{m[2]} {m[3]} {m[4]}", format="%B %d %Y")
    if m[1] and date.day_name().lower() != m[1].lower():
        return None
    return date


# ---------------------------------------------------------------- table

def build_event_table(paths: RawPaths) -> pd.DataFrame:
    """All events, one row each: event, date, label, used, date_source, file_date, source."""
    frames = []
    for year, page in sorted(paths.fomc_historical.items()):
        frames.append(parse_fomc_historical(page.read_text(errors="replace"))
                      .assign(source=f"fomc/{page.name}"))
    frames.append(parse_fomc_calendar(paths.fomc_calendar.read_text(errors="replace"))
                  .assign(source=f"fomc/{paths.fomc_calendar.name}"))
    fomc = pd.concat(frames, ignore_index=True).assign(event="fomc", date_source="fed calendar")
    fomc = fomc[fomc["date"].dt.year >= FOMC_FIRST_YEAR]

    bls = []
    for event, page in paths.bls_archive.items():
        series = page.name.split("_")[0]
        for r in parse_bls_archive(page.read_text(errors="replace"), series).itertuples():
            doc = paths.bls_docs[event] / Path(r.doc_path).name
            emb = embargo_date(doc.read_text(errors="replace")) if doc.exists() else None
            bls.append({"event": event, "date": emb if emb is not None else r.file_date,
                        "label": f"{r.ref_month} {series}", "used": True,
                        "date_source": "embargo line" if emb is not None else "file name",
                        "file_date": r.file_date, "source": f"bls/{series}/{doc.name}"})

    table = pd.concat([fomc, pd.DataFrame(bls)], ignore_index=True)
    cols = ["event", "date", "label", "used", "date_source", "file_date", "source"]
    return table[cols].sort_values(["event", "date", "label"]).reset_index(drop=True)


def coverage_start(table: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """First day each flag is defined. FOMC: the whole of 1994 is covered by its calendar.
    BLS: the archive starts with the first release it holds, so coverage starts there."""
    used = table[table["used"]]
    start = {e: used.loc[used["event"] == e, "date"].min() for e in ("cpi", "jobs")}
    start["fomc"] = pd.Timestamp(FOMC_FIRST_YEAR, 1, 1)
    return start
