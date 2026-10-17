# volatility-managed-portfolio
Test and Implement volatility based strategy on managing portfolios risk

## Phase 1: Data and Set up

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


## Phase 2: Variance and EDA

Variance:


Tests:
- 

- 

- 

- Test 4: to measure if its rewarding more if we hold less on the assets on risky times, if not then it would just be better for us to buy&hold. How it works:
1. Each month, score how volatile and record the returns and variance of each assets
2. Sort each pile of months from calmest to wildest and for each pile, compute their avg returns, variance, then the ratio
3. Then managed the return as for each month, multiply the next return by a weight which is c + last month variance. After a wild month, hold less, after a calm month, hold more
4. The constant c is set in order for managed and original have the asme volatility and allow them to be comparable, and not by their values
5. Compare managed with original: with regression managed: a + B * original then t-test to minimize chance of luck

## Phase 3: Assets Screening

Assets Screening with multiple filters to keep only assets with chance of success.

Filters:
- Predictable Risks: from test on phase 2, test 1 (fail: nothing to forecast). Result: all predictable

- Moving Risks: SD of log RV to measure how much variance moves (the idea is that if the risks is not moving with time, which is flat risk, then our whole idea of strategy would be useless as it would just be the same as buy&hold). Result: NO asset is flat

- Exploitability for Strats: risk to reward ratio on how our assets can be exploited, where the reward must fall when the risk rises. Such that our strategy of reducing out position weight can be reduced when its getting risky. If the risk and reward is one on one, then there is no use to adjust our position based on the risk. Test from Phase 2, test 4 (the test measures how rewarding is it from risky times compared to calm times). Result: Only 3 assets: Market, Momentum, and BAB.

- Stability: to ensure that the Strats is stable and applicable in all eras of time such that the Strats is still viable and not just based on specific era. Test: using the phase 2 test 4, implemented on splitting the history into 3 eras, then check if the a variable stays > 0 as a consistency check that the Strats is not a single era accident

- 

