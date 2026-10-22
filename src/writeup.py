"""
Track D: the figures of the Phases 1-9 report
=============================================
    python -m src.writeup      writes reports/figures/report_gains.png and report_holdout.png

Both read saved outputs only (reports/phase7_*.csv, reports/phase9_results.csv, data/processed/phase9_runs.parquet).
Nothing here opens the holdout or computes a new result.

    gains()     every pre-registered Sharpe gain, development beside holdout, with its 90% interval (the one-sided 95%
                bounds the readings use)
    holdout()   growth of 1 relative to cash over the holdout: the primary and SPY alone, timed against buy-and-hold
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, REPORTS_DIR

OUT = REPORTS_DIR / "figures"
# The reference palette (light surface): two categorical slots, validated together; ink and chrome tokens.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def gain_rows() -> pd.DataFrame:
    """(label, sample, gain, lower, upper) for each registered book, from the saved tables."""
    r = pd.read_csv(REPORTS_DIR / "phase7_results.csv")
    t = pd.read_csv(REPORTS_DIR / "phase7_tracks.csv").set_index("book")
    h = pd.read_csv(REPORTS_DIR / "phase9_results.csv")
    prim = r[r["row"].astype(str).str.startswith("primary")].iloc[0]
    best = r[(r.model == "M0") & (r.rule == "down_vol") & (r.mix == "shrunk_mv") & (r.freq == "weekly")].iloc[0]
    yw = pd.read_csv(REPORTS_DIR / "phase8_year_weighted.csv")
    dev_c = yw[yw["rule"].str.startswith("(c)")].iloc[0]                    # Entry 11 (information)
    hold = lambda b, s: h[(h.book == b) & (h.statistic == s)].iloc[0]
    rows = [("Primary, statistic (a)", "development", prim.gain, prim.lower, prim.upper),
            ("Primary, statistic (a)", "holdout", *hold("primary", "(a)")[["stat_gain", "stat_lower", "stat_upper"]]),
            ("Primary, statistic (c): the final test", "development", dev_c.gain, dev_c.lower, dev_c.upper),
            ("Primary, statistic (c): the final test", "holdout", *hold("primary", "(c)")[["stat_gain", "stat_lower", "stat_upper"]]),
            ("Best development variant", "development", best.gain, best.lower, best.upper),
            ("Best development variant", "holdout", *hold("best", "(c)")[["stat_gain", "stat_lower", "stat_upper"]]),
            ("SPY alone, after costs", "development", t.loc["spy", "gain"], t.loc["spy", "lower"], t.loc["spy", "upper"]),
            ("SPY alone, after costs", "holdout", *hold("spy", "(a)")[["stat_gain", "stat_lower", "stat_upper"]]),
            ("Fixed book (information)", "development", t.loc["fixed", "gain"], t.loc["fixed", "lower"], t.loc["fixed", "upper"]),
            ("Fixed book (information)", "holdout", *hold("fixed", "(a)")[["stat_gain", "stat_lower", "stat_upper"]])]
    return pd.DataFrame(rows, columns=["label", "sample", "gain", "lower", "upper"])


def gains() -> None:
    d = gain_rows()
    labels = list(dict.fromkeys(d["label"]))
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for i, lab in enumerate(labels):
        for sample, color, dy in (("development", BLUE, -0.17), ("holdout", ORANGE, 0.17)):
            r = d[(d.label == lab) & (d["sample"] == sample)].iloc[0]
            y = i + dy
            ax.plot([r.lower, r.upper], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(r.gain, y, "o", markersize=7, color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9)
    ax.set_xlabel("Sharpe gain over buy-and-hold, a year", color=INK_2, fontsize=9)
    ax.set_xlim(-0.55, 0.65)
    handles = [plt.Line2D([], [], color=c, marker="o", markersize=7, markeredgecolor=SURFACE, linewidth=2, label=l)
               for c, l in ((BLUE, "Development (1926-2018 walk-forward)"), (ORANGE, "Holdout (2019-2026, opened once)"))]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=2, frameon=False, fontsize=9,
              labelcolor=INK_2, handletextpad=0.5, columnspacing=1.6)
    fig.text(0.01, 0.985, "Every estimate, with its 90% interval", color=INK, fontsize=11, weight="semibold", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "report_gains.png", facecolor=SURFACE)
    plt.close(fig)


def holdout() -> None:
    runs = pd.read_parquet(PROCESSED_DIR / "phase9_runs.parquet")
    panels = (("primary", "Primary (paper factors, no costs)"), ("spy", "SPY alone (after costs)"))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, (book, title) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        m = runs[(runs.book == book) & runs.holdout].copy()
        m["month"] = pd.PeriodIndex(m["month"], freq="M")
        m = m.set_index("month").sort_index()
        start = m.index[0] - 1
        x = [p.to_timestamp(how="end") for p in [start, *m.index]]
        ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-03-31"), color=GRID, linewidth=0, zorder=0)
        ends = []
        for col, color, name in (("excess", BLUE, "Timed"), ("bench_excess", ORANGE, "Buy-and-hold")):
            g = np.r_[1.0, np.cumprod(1 + m[col].to_numpy())]
            ax.plot(x, g, color=color, linewidth=2, solid_joinstyle="round", zorder=2)
            ax.plot(x[-1], g[-1], "o", markersize=5, color=color, markeredgewidth=0, zorder=3)   # no ring: it would cut the last month
            ends.append((g[-1], name))
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        (y1, n1), (y2, n2) = sorted(ends)
        if y2 - y1 < gap:
            y1, y2 = (y1 + y2) / 2 - gap / 2, (y1 + y2) / 2 + gap / 2
        for y, (v, name) in zip((y1, y2), sorted(ends)):
            ax.annotate(f"{name} {v:.2f}", (x[-1], y), xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=INK_2)
        ax.set_title(title, loc="left", color=INK, fontsize=10, weight="semibold")
        ax.margins(x=0.02)
        ax.set_xlim(right=x[-1] + pd.Timedelta(days=640))               # room for the end labels, no tick past the data
        years = pd.date_range("2019-01-01", x[-1], freq="YS")
        ax.set_xticks(years, [str(d.year) for d in years])
        ax.text(pd.Timestamp("2020-03-01"), hi - 0.02 * (hi - lo), "2020 crash", color=MUTED, fontsize=8, ha="center", va="top")
    axes[0].set_ylabel("Growth of 1, relative to cash", color=INK_2, fontsize=9)
    handles = [plt.Line2D([], [], color=c, linewidth=2, label=l) for c, l in ((BLUE, "Timed (frozen rule)"), (ORANGE, "Buy-and-hold"))]
    fig.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)
    fig.text(0.01, 0.985, "The holdout, January 2019 onward", color=INK, fontsize=11, weight="semibold", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "report_holdout.png", facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gains()
    holdout()
    print(f"wrote {OUT / 'report_gains.png'} and {OUT / 'report_holdout.png'}")


if __name__ == "__main__":
    main()
