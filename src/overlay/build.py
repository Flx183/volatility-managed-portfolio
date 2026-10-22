"""
Build Track A's development steps
=================================
    python -m src.overlay.build --power   the 36-month power check (Entries 20 and 25), the first development step:
                                          A1 on the twin; writes reports/phase10_tracka_power.csv

The power check decides the primary metric before anything else of Track A is run. If volatility stability's
power at 36 months is at least 80%, it stays the primary metric; below that, the Sortino gain takes its place,
with the 0.16 bound (TRACK_A["fallback"]).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from src.config import REPORTS_DIR
from src.data.load import load_dev
from src.engine import execution
from src.overlay import OPEN, TRACK_A, metrics, require_decided, sizing
from src.walkforward import runs, tracks

POWER_BAR = 0.80                          # Entry 20: below this, the Sortino gain becomes the primary metric
AT_MONTHS = 36                            # Holdout C's formal test, 36 months after the freeze


def twin_a1(panel: pd.DataFrame) -> tuple:
    """A1 on the twin: the market factor at SPY's costs, M2, A0's months (c_t from 120 pairs), the cap of 1.5, and
    σ̂_y from the twin's last 10 years at each December year-end."""
    s = tracks.specs(panel)["market"]
    book = s["book"]
    periods = execution.holding_periods(book, s["costs"])
    res = sizing.run(runs.prepare(book, s["forecaster"]), periods, "A1", sigma=sizing.sigma_trailing(book),
                     max_size=TRACK_A["cap"], min_pairs=s["min_pairs"])
    return book, res, periods


def power_check(panel: pd.DataFrame) -> pd.DataFrame:
    book, res, periods = twin_a1(panel)
    months = runs.scored(res, periods).index
    t = metrics.vol_stability_test(metrics.monthly_vol(book, res, periods, months))
    power = metrics.power_at(t["reduction"], t["se"], t["months"], AT_MONTHS)
    primary = "vol_stability" if power >= POWER_BAR else "sortino"
    return pd.DataFrame([{"variant": "A1", "book": "twin", "first": str(months[0]), "last": str(months[-1]), **t,
                          "power_at_36": power, "power_bar": POWER_BAR, "primary_metric": primary,
                          "registered_primary": OPEN["primary_metric"]}])


def main(power: bool) -> None:
    if not power:
        raise SystemExit("only --power is built: the power check is Track A's first development step")
    require_decided()
    t0 = time.time()
    T = power_check(load_dev())
    T.to_csv(REPORTS_DIR / "phase10_tracka_power.csv", index=False)
    r = T.iloc[0]
    print(f"A1 on the twin, {r['months']} months: power at 36 months {r['power_at_36']:.1%}; "
          f"primary metric: {r['primary_metric']} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", action="store_true", help="run the 36-month power check")
    main(ap.parse_args().power)
