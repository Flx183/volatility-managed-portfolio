import pandas as pd
import pytest

from src.data import events


@pytest.mark.parametrize("label, year, expected", [
    ("January 27-28", 2015, "2015-01-28"),
    ("August 5", 2008, "2008-08-05"),
    ("January 31-February 1", 1995, "1995-02-01"),
    ("Jan/Feb 31-1", 2017, "2017-02-01"),
    ("Apr/May 30-1", 2024, "2024-05-01"),
    ("September 16-17*", 2025, "2025-09-17"),
])
def test_meeting_end(label, year, expected):
    assert events.meeting_end(label, year) == pd.Timestamp(expected)


def _panel(heading, body="<p><a href='/x.htm'>Statement</a></p>"):
    return f'<div class="panel-heading"><h5>{heading}</h5></div><div>{body}</div>'


def test_historical_keeps_only_scheduled_policy_meetings():
    page = "".join([
        _panel("January 27-28 Meeting - 2015"),
        _panel("October 16 (unscheduled) - 2013"),
        _panel("January 21 Conference Call - 2008"),
        _panel("March 17-18 (cancelled) Meeting - 2020"),
        _panel("September 15 Meeting - 2003", body="<p><a href='/a.pdf'>Agenda</a></p>"),
        _panel("August 16 Meeting - 1994", body="<p>Minutes</p>"),   # pre-1999: no statement needed
        '<h5 class="panel-heading panel-heading--shaded">March 2 (unscheduled) Meeting - 2020</h5><div></div>',
    ])
    df = events.parse_fomc_historical(page)
    used = df.loc[df["used"], "date"].dt.strftime("%Y-%m-%d").tolist()
    assert used == ["2015-01-28", "1994-08-16"]
    assert len(df) == 7


def test_calendar_page_skips_notation_votes():
    row = ('<div class="fomc-meeting__month col"><strong>{m}</strong></div>'
           '<div class="fomc-meeting__date col">{d}</div>')
    page = ('<h4><a id="1">2024 FOMC Meetings</a></h4>'
            + row.format(m="Apr/May", d="30-1") + row.format(m="June", d="11-12*")
            + '<h4><a id="2">2025 FOMC Meetings</a></h4>'
            + row.format(m="August", d="22 (notation vote)"))
    df = events.parse_fomc_calendar(page)
    assert df.loc[df["used"], "date"].tolist() == [pd.Timestamp("2024-05-01"), pd.Timestamp("2024-06-12")]
    assert not df.loc[df["date"] == "2025-08-22", "used"].item()
    assert events.fomc_calendar_years(page) == [2024, 2025]


def test_bls_archive_parsing():
    page = """
    <ul><!-- <li><a href="/news.release/archives/empsit_11062026.htm">October 2026 Employment Situation</a></li> -->
    <li><a href="https://www.bls.gov/news.release/archives/empsit_08022019.htm">July 2019 Employment Situation</a>
        (<a href="https://www.bls.gov/news.release/archives/empsit_08022019.pdf">PDF</a>)</li>
    <li>October 2025 Employment Situation &ndash; Not published because of 2025 lapse</li>
    <li>December 1995 Employment Situation (<a href="/news.release/history/empsit_011996.txt">TXT</a>)</li>
    <li>January 1995 Employment Situation (<a href="/news.release/archives/empsit_02031995.pdf">PDF</a>)</li>
    </ul>"""
    df = events.parse_bls_archive(page, "empsit")
    assert df["ref_month"].astype(str).tolist() == ["1995-01", "1995-12", "2019-07"]
    assert df["file_date"].tolist() == [pd.Timestamp("1995-02-03"), pd.Timestamp("1996-01-19"),
                                        pd.Timestamp("2019-08-02")]
    assert df["doc_path"].iloc[2].endswith(".htm")          # htm preferred over pdf


def test_embargo_date_from_old_two_column_header():
    doc = """Media contact:           691-5902  embargoed until 8:30 A.M. (EST),
                                     Friday, January 7, 2000.
                 THE EMPLOYMENT SITUATION:  DECEMBER 1999"""
    assert events.embargo_date(doc) == pd.Timestamp("2000-01-07")


def test_embargo_date_from_html_and_weekday_guard():
    html = ("<script>var x='Monday, May 1, 2000';</script><p>Transmission of material in this release "
            "is embargoed until</p><p>8:30 a.m. (ET) Friday, September 11, 2026</p>")
    assert events.embargo_date(html) == pd.Timestamp("2026-09-11")
    assert events.embargo_date("embargoed until 8:30 a.m. Monday, September 11, 2026") is None
    assert events.embargo_date("no notice here, Friday, September 11, 2026") is None


def test_embargo_date_without_weekday_or_space_before_year():
    assert (events.embargo_date("is embargoed until 8:30 a.m. (EST) January 11, 2019 USDL-19-0018")
            == pd.Timestamp("2019-01-11"))
    assert (events.embargo_date("RELEASE IS EMBARGOED UNTIL 8:30 A.M. (EDT) Friday, October 16,1998")
            == pd.Timestamp("1998-10-16"))


def test_for_release_header_and_reissue_use_the_day_the_news_came_out():
    doc = "For release: 1:30 P.M. (EST)   Media contact: 606-5902   Thursday, November 5, 1998"
    assert events.embargo_date(doc) == pd.Timestamp("1998-11-05")
    reissue = ("(Replaces USDL 02-669, issued December 6, 2002.)  For release: Immediate "
               "Monday, December 9, 2002.  THE EMPLOYMENT SITUATION: NOVEMBER 2002")
    assert events.embargo_date(reissue) == pd.Timestamp("2002-12-06")


def test_embargo_weekday_typo_falls_back():
    # The January 1997 CPI release prints 'Tuesday, January 14, 1996' (a Sunday).
    assert events.embargo_date("EMBARGOED UNTIL 8:30 A.M. (EST) Tuesday, January 14, 1996") is None
