# volatility-managed-portfolio
Test and Implement volatility based strategy on managing portfolios risk

## Phase 1: data

```bash
pip install -r requirement.txt
python -m src.data.build      # snapshot sources, build panels, run the pass rule
python -m pytest              # unit tests
```

The build writes `reports/phase1_data_report.md` (pass rule, coverage, sources) and
`data/events/event_dates.csv` (every FOMC, CPI and jobs-report date, with where it came from).

```python
from src.data.load import load_dev, load_etf_ohlc_dev

panel = load_dev()   # one row per trading day, 1926-07-01 .. 2018-12-31
```

`panel` columns: factor excess returns `mkt smb hml mom rmw cma bab`, ETF excess returns
`spy mtum qual usmv`, `rf`, `vix` and event flags `fomc cpi jobs`.

The holdout (2019 onward) sits in separate files, sealed by a content hash in
`data/holdout_lock.json`. It is opened once, in Phase 9, with
`load_holdout(confirm="phase 9 final test")`, and every opening is logged in
`data/holdout_access_log.csv`.
