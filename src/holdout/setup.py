"""
Phase 9, setup: the joined panel, the books and the crisis spans
================================================================
    join(dev, holdout)                 one table: the development rows, then the holdout's (no overlap allowed)
    complete_end(panel, cols)          the last trading day of the last month in which every column in cols has
                                       a return on every trading day
    cut(panel, ohlc, end)              the panel and the ETF table up to that day
    screen(paper, ohlc, jobs)          Phase 3's screen at every December year-end (Phase 7's screened.screen_all,
                                       without the data's last day, which is not a year-end) and its survivors
    specs(paper, etf_panel, rec, start)  every Phase 9 book with its frozen forecaster, power, costs and c_t minimum
    prepare(spec), run(spec, ...)      a book through Phase 7's walk-forward runner, with the cap or costs changed
                                       for the crisis questions
    span(panel, peak_from, trough_year)  a holdout crisis: the market's trough (its lowest close in the year) and
                                       peak (its highest close from peak_from up to the trough), and the window
                                       from 6 months before the peak to 12 months after the trough

Every rule is in HOLDOUT_RULES and EPISODES (PREREGISTRATION Entry 13).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import engine as e
from src import models as md
from src import portfolio as pf
from src import screen as sc
from src.engine import execution
from src.engine.data import MONTH
from src.holdout import HOLDOUT_RULES
from src.robustness import ROBUST_RULES, checks
from src.walkforward import FREQS, RULES, runs, screened, tracks

MAX_SIZE = e.ENGINE_RULES["max_size"]
KEEP = object()                                       # run(): keep the spec's own cap or costs


def join(dev: pd.DataFrame, holdout: pd.DataFrame, date: str | None = None) -> pd.DataFrame:
    """The development rows, then the holdout's: the daily panel (date=None, indexed by day) or the long ETF
    table (date="date")."""
    if list(dev.columns) != list(holdout.columns):
        raise ValueError("the development and holdout tables have different columns")
    a, b = (dev.index, holdout.index) if date is None else (dev[date], holdout[date])
    if b.min() <= a.max():
        raise ValueError(f"the holdout starts on {b.min()}, not after the development sample ({a.max()})")
    out = pd.concat([dev, holdout])
    return out if date is None else out.reset_index(drop=True)


def complete_end(panel: pd.DataFrame, cols: list[str]) -> pd.Timestamp:
    """The last trading day of the last month whose every trading day has a return in every column of cols.
    AQR's BAB file ends a month before the others (Phase 1's report), so the paper books end there."""
    month = panel.index.to_period(MONTH)
    full = panel[cols].notna().all(axis=1).groupby(month).all()
    last = full.index[full.to_numpy()][-1]
    return panel.index[month == last].max()


def cut(panel: pd.DataFrame, ohlc: pd.DataFrame, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    return panel.loc[:end], ohlc[ohlc["date"] <= end].reset_index(drop=True)


def december_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The year-ends of Phase 3's screen that close a December: the data's last day is not a year-end."""
    ends = sc.year_ends(index)
    return ends[ends.month == 12]


def screen(paper: pd.DataFrame, ohlc: pd.DataFrame, jobs: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """screen(panel, ohlc, asof) at every December year-end, each reading data up to its date only (Phase 7's
    screened.screen_all on those dates), and the factors held after each (screened.survivors)."""
    dates = december_ends(paper.index)
    tables = Parallel(n_jobs=jobs)(delayed(sc.screen)(paper, ohlc, d) for d in dates)
    table = pd.concat(tables, keys=dates, names=["asof", "factor"])
    return table, screened.survivors(table)


def from_start(book: e.Book, start: pd.Timestamp) -> e.Book:
    """The book's holding periods from the one holding `start` on: the fixed book (information) is run on the
    holdout only, so only the recipes held there need forecasts. A recipe's c_t and forecast come from its
    mix's whole history, so the decisions are those of the full book."""
    keep = book.positions.index.end_time >= start
    return e.Book(book.returns, book.rf, book.positions.loc[keep])


def specs(paper: pd.DataFrame, etf_panel: pd.DataFrame, rec: pd.DataFrame, start: pd.Timestamp) -> dict[str, dict]:
    """Each Phase 9 book: book, forecaster, power, costs, c_t minimum, the mix behind it (for option (c)), the
    forecast's horizon in trading days (to read σ̂ as a volatility) and its frequency. The paper books use the
    paper panel; the ETF track the ETF panel, which runs a month longer."""
    model, rule, mix, freq = HOLDOUT_RULES["primary"]
    b_model, b_rule, b_mix, b_freq = HOLDOUT_RULES["best"]
    monthly = md.forecasters(paper, False)                     # M2 and M3 do not read `down`
    weekly = md.forecasters(paper, False, FREQS[b_freq])
    fixed = from_start(checks.book(paper, checks.recipes(paper, "fixed")), start)
    out = {
        "primary": {"book": screened.book(paper, rec, mix, FREQS[freq]), "forecaster": monthly[model],
                    "power": RULES[rule][0], "costs": e.COSTS["paper"], "min_pairs": runs.MIN_PAIRS[FREQS[freq]],
                    "mix": mix, "horizon": 22},
        "best": {"book": screened.book(paper, rec, b_mix, FREQS[b_freq]),
                 "forecaster": runs.downside(weekly[b_model], FREQS[b_freq]) if RULES[b_rule][1] else weekly[b_model],
                 "power": RULES[b_rule][0], "costs": e.COSTS["paper"], "min_pairs": runs.MIN_PAIRS[FREQS[b_freq]],
                 "mix": b_mix, "horizon": 5},
        "spy": {"book": e.Book.single(etf_panel["spy"], etf_panel["rf"], "spy"), "forecaster": tracks.etf_garch(False),
                "power": 1.0, "costs": e.COSTS["etf"], "min_pairs": tracks.ETF_MIN_PAIRS, "mix": None, "horizon": 22},
        "etf": {"book": e.data.from_phase4(etf_panel, pf.recipes(etf_panel, "etf"), pf.PRIMARY[1]),
                "forecaster": tracks.etf_garch(True), "power": 1.0, "costs": e.COSTS["etf"],
                "min_pairs": tracks.ETF_MIN_PAIRS, "mix": None, "horizon": 22},
        "fixed": {"book": fixed, "forecaster": monthly["M3"], "power": 1.0, "costs": e.COSTS["paper"],
                  "min_pairs": runs.MIN_PAIRS[MONTH], "mix": None, "horizon": 22},
    }
    return out


def prepare(spec: dict) -> list:
    """The engine's stages 1-2 on every segment of the book (each recipe's forecast over its mix's history)."""
    return runs.prepare(spec["book"], spec["forecaster"])


def run(spec: dict, prepared: list, max_size=KEEP, costs=KEEP) -> tuple[e.Book, pd.DataFrame, execution.Periods]:
    """One book through Phase 7's runner: (book, res, periods). max_size and costs change the spec's own (the
    crisis questions: no cap, a cap of 1.0, 30 bp a trade)."""
    cap = MAX_SIZE if max_size is KEEP else max_size
    periods = execution.holding_periods(spec["book"], spec["costs"] if costs is KEEP else costs)
    res = runs.run(prepared, spec["power"], periods, max_size=cap, min_pairs=spec["min_pairs"])
    return spec["book"], res, periods


def kappa_everywhere(book: e.Book, kappa: float) -> e.Costs:
    """κ one-way on every instrument the book holds, in both runs (Entry 8's cost question)."""
    return e.Costs(kappa={c: kappa for c in book.positions.columns})


def span(panel: pd.DataFrame, peak_from: str, trough_year: str) -> dict:
    """crises.span's dict for a holdout episode, from the market's total-return level (excess return plus rf)."""
    level = (1 + panel["mkt"] + panel["rf"]).loc[peak_from:f"{trough_year}-12-31"].cumprod()
    trough = level.loc[trough_year].idxmin()
    peak = level.loc[:trough].idxmax()
    start = peak - pd.DateOffset(months=ROBUST_RULES["lead_months"])
    end = trough + pd.DateOffset(months=ROBUST_RULES["after_months"])
    return {"peak": peak, "trough": trough, "start": start, "end": end, "market_fall": level[trough] / level[peak] - 1,
            "months": pd.period_range(start.to_period(MONTH), end.to_period(MONTH), freq=MONTH)}


def lo_se(x: pd.Series) -> float:
    """Lo's (2002) standard error of an annual Sharpe ratio from monthly excess returns, for returns that are
    independent over time: √((1 + SR_m² / 2) / T), with SR_m the monthly Sharpe ratio, times √12."""
    sr = x.mean() / x.std()
    return float(np.sqrt((1 + 0.5 * sr ** 2) / len(x)) * np.sqrt(12))


def lo_example(sharpe: float = 0.5, months: int = 92) -> dict:
    """The plan's worked example: the standard error of an annual Sharpe ratio of 0.5 over 92 months."""
    sr = sharpe / np.sqrt(12)
    se_m = np.sqrt((1 + 0.5 * sr ** 2) / months)
    return {"sharpe": sharpe, "months": months, "monthly_sharpe": sr, "se_monthly": se_m, "se_annual": se_m * np.sqrt(12),
            "half_width_95": 1.96 * se_m * np.sqrt(12)}

