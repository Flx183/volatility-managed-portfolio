# Pre-Registration: Filtering, Implementing, Optimizing managed Portfolio

This document fixes the hypotheses, evaluation procedure, and success threshold before the experiment is run or any steps after. It exists  so the verdicts in this repository cannot be reverse-engineered from the
data. Nothing below may be edited after results are produced; if the apparatus
turns out to be wrong, fix it in a new, dated entry and re-run, rather than
moving a goalpost.

## How this file works

- Entries are appended in date order. Once a result an entry governs exists, the entry is not edited; a correction is a new dated entry that names what it replaces.
- Each entry says which rules were fixed **before** the results they govern and which were written **after** a first run, with the date and the reason.

---

## Entry 1 — 2026-10-18: Phases 1–4 on record, and what is fixed for the final test

**Read this first.** This file held no rules until this entry, and Phases 1 to 4 had already been run. For those phases, this entry is a **record, not a pre-registration**. It transcribes the rules each phase's code holds and each phase report says were fixed before the first run, plus every change made after a first run. The commit history shows when rules were committed, not whether they were written before the first run; for that order, the phase reports are the evidence.

This entry pre-registers, in the full sense, the holdout and the final test (Sections 6 and 7). The holdout has never been opened.

### 1. Question and hypotheses

**Question.** Does scaling a fixed, diversified factor portfolio by the inverse of its own forecast variance beat holding the same portfolio unmanaged? Every choice is made from past data only, and the comparison is out of sample and after costs.

- **H1.** The volatility-managed mix has a higher Sharpe ratio than the buy-and-hold mix in the holdout, after costs.
- **H0.** It does not. This is the skeptic's prior. Cederburg et al. (2020) find that volatility-managed portfolios do not reliably beat their unmanaged versions in real-time, out-of-sample comparisons. They also find that the gains shrink once the Fama-French factors are in the opportunity set (momentum 0.44 to 0.20, BAB 0.30 to 0.20, market −0.04 to −0.11).

**Benchmark.** The buy-and-hold mix: the same assets and the same mix weights, held at a constant size. Its Sharpe ratio on the development sample is 1.14–1.17 (Phase 4), so the bar is high.

The statistic, test and cutoff that decide between H1 and H0 are not yet fixed (Section 7, item 1).

### 2. Data and the sealed holdout (Phase 1, run 2026-10-07)

| Item | Rule |
|---|---|
| Sources | Kenneth French daily factors (market, size, value, profitability, investment, momentum); AQR betting-against-beta (BAB), daily; Yahoo SPY, MTUM, QUAL, USMV; CBOE VIX; FOMC, CPI and jobs-report dates. Every raw file's URL, download time and SHA-256 is in `data/raw_manifest.json`. |
| Development sample | 1926-07-01 to 2018-12-31. All analysis, choices and tuning use only this sample. |
| Holdout | 2019-01-01 onward (data to 2026-08-31), sealed on 2026-10-07 by content hash in `data/holdout_lock.json`. It is opened only through `load_holdout(confirm="phase 9 final test")`, which checks the hash and logs every opening in `data/holdout_access_log.csv`. As of this entry that log does not exist: the holdout has never been opened. Phase 1's checks read only dates, completeness and violation counts from it. |
| ETFs | 5–7 years of history before 2019, under the 10-year initial window, so they are never tuned on. Every rule and cutoff comes from the paper factors. |
| Pass rule | Nine gating checks in `src/data/checks.py`: one trading calendar, no missing days, SPY against an independent S&P 500 source (Damodaran), same-day alignment, known extreme days, Yahoo internal consistency, value bounds, event counts, and a non-overlapping split. **Passed** on 2026-10-07 (`reports/phase1_data_report.md`). |

Rebuilding or resealing the holdout (`--refresh`, `--relock`) would change it. Any such change needs its own dated entry before anything reads the holdout.

### 3. Phase 2 — Variance diagnostics (`src/variance.py`, `reports/phase2_variance.ipynb`)

The thresholds are the `RULES` constants. The report states they were written before the first test ran, and they have not changed.

| Test | Pass rule |
|---|---|
| 1 Predictable (the gate) | Ljung-Box on r² rejects at 1% at 20 **and** 60 lags, **and** the monthly AR(1) slope of realized variance, φ, is above 0.5 |
| 2 Memory | weekly and monthly HAR coefficients both > 0 with Newey-West t > 1.96 |
| 3 Bad days | the extra effect of a down day on future variance, δ, is > 0 with Newey-West t > 1.96 |
| 4 Reward vs risk | next month's mean/variance is lower in the wildest quintile of this month's variance than in the calmest, with rank correlation ≤ −0.7, **and** the spanning regression gives α > 0 with White t > 1.96 |
| 5 Events / VIX | events: mean r² ≥ 1.2× other days with t > 1.96. VIX: t > 1.96 and R² gain ≥ 0.01 |
| 6 Eras (1926–1962, 1963–1990, 1991–2018) | every era has the full-sample sign and at least ⅓ of its size |
| 7 Shape | log RV has abs skewness < 0.5 and kurtosis < 4 |
| 8 Collinearity | a VIF above 10 is severe |
| 9 Curvature | the quadratic term, or the slope shift in the top 20% of months, has abs Newey-West t > 1.96 |
| 10 History and stability | A: history helps until a stage adds less than 0.5 point of out-of-sample R². B: rolling 10-year φ leaves its 95% band in more than 20% of windows, or CUSUM rejects a stable mean at 5% |

**Conventions.** Significance means abs t > 1.96. Newey-West uses 44 lags for daily regressions with 22-day overlapping targets and 6 lags for monthly ones. White standard errors are HC0.

**Added after results.** Two explanation checks were added after the first run: October 1987 in Test 6, and the August 2015 ETF dislocation. Both are labelled in the report, and neither changes a verdict.

**Outcome.** Test 1 passes for all 8 assets. Test 4 passes for market, momentum, BAB and the equal-weight placeholder.

### 4. Phase 3 — Asset screening (`src/screen.py`, `reports/phase3_screening.ipynb`)

`screen(panel, ohlc, asof)` reads only data up to `asof`. The cutoffs are in `SCREEN_RULES`.

| # | Filter | Pass rule | Applies to |
|---|---|---|---|
| 1 | Predictable risk | Phase 2 Test 1 | factor |
| 2 | Risk that moves | standard deviation of log RV; ranked, no cutoff (tie-breaker) | factor |
| 3 | Exploitable link | Phase 2 Test 4 | factor |
| 4 | Systematic risk | ETF minus β × market (β from data up to the as-of date), regressed on market, size, value, momentum and BAB: R² > 0.5. SPY is tested unhedged | ETF |
| 5 | Tradeable | median daily dollar volume over the last 252 trading days above $50 million | ETF |
| 6 | Stable | spanning α > 0 in each of three equal thirds of the history up to the as-of date | factor |

**Fixed choices.**

- A factor needs 120 months of history and an ETF 252 trading days.
- Windows are expanding.
- The screen runs only at month-ends.
- The ETF stand-ins are SPY (market), MTUM (momentum), QUAL (profitability) and USMV (BAB). Size, value and investment have none.
- A factor survives only if it passes filters 1, 3, 4, 5 and 6.

**Changes after a first run.**

- **2026-10-17, filter 4.** The rule was: raw ETF on market, size, value and momentum, R² > 0.7. It could not fail: every ETF scored 0.82–0.96, and the market alone explained 0.78–0.95. It was replaced by the hedged-tilt test above, with the 0.5 cutoff fixed before the re-run. The new rule decides two ETFs: MTUM passes (0.53) and USMV fails (0.38).
- **Not applied: filter 5's spread half** (bid-ask spread below 5 bp). There is no quote data, and the high-low proxy reads 7–15 bp for SPY. Spreads are to be verified in the cost phase.

**Outcome (full development sample).** The survivors are market (SPY) and momentum (MTUM). BAB passes every factor filter, but USMV fails filter 4, so BAB has no tradeable stand-in.

### 5. Phase 4 — Portfolio construction (`src/portfolio.py`, `reports/phase4_portfolio.ipynb`)

The rules are in `PORTFOLIO_RULES`, `PRIMARY` and `TRACKS`, written into the code before the first run on 2026-10-17.

| Step | Rule |
|---|---|
| 1 Duplicates | A pair whose monthly returns have ρ > +0.7, on all months **or** on crisis months (the 10% of months with the highest market variance), is a duplicate. The asset with the shorter history is dropped (if the histories tie, the weaker Phase 2 Test 4 t-stat). The check uses all history up to the as-of date and needs at least 120 common months |
| 2 Weights | Four mix versions: equal; inverse variance; inverse volatility; shrunk mean-variance. The last averages 1/N with the normalized Σ̂⁻¹μ̂, using a Ledoit-Wolf Σ̂ with negatives set to zero (1/N if every element is negative). Estimated from the last 10 years of daily returns (all history if shorter), set at each year-end, and held for the next calendar year. The paper tracks need 120 months of history, the ETF track 12 |
| 3 Combine, then time | p = Σ wᵢ rᵢ each day. Size for a month = min(c / last month's realized variance of p, 1.5). c is set with the recipe so that, over the estimation window, the uncapped timed mix and the untimed mix have the same variance. Exposure to asset i = size × wᵢ |
| Tracks | **Paper**: market, momentum, BAB. **ETF**: SPY plus MTUM's tilt (MTUM − β × SPY, β from the estimation window). **Paper market + momentum**: the ETF track's long-history twin |
| Primary | the paper track with inverse-volatility weights |
| Pass rule | Phase 2 Tests 1 and 4 pass on the primary p. Level φ is the gate, with log φ reported beside it. The test is judged only with at least 120 months of p. If the primary fails, it stays the primary and the failure is reported |

**Fixed choices.**

- Every estimate (σ̂, Σ̂, μ̂, β) uses daily returns.
- Crisis months are ranked on the market's variance.
- The asset list is Phase 3's full-sample survivor list; Phase 7 replaces it with the real-time screen.
- Until Phase 6 chooses a model, the variance forecast is last month's realized variance.

**Changes after the first run.** Both were made on 2026-10-17, and neither changes a result.

1. **Minimum history for the duplicate check.** The first run judged the ETF pair on 20–68 months, so its crisis set had 2–7 months. With two points a correlation is ±1 by construction, and the MTUM tilt was dropped. The check now needs 120 common months, Phase 3's existing minimum.
2. **Duplicates on ρ > +0.7 instead of |ρ| > 0.7.** A duplicate is the same bet, and a strongly negative correlation is a hedge. This change was made after seeing the market–momentum crisis correlation reach −0.69 in 1948. No pair crossed either version of the rule at any year-end.

**Calls made on 2026-10-17, after the first run.**

| Decision | Call | Reason |
|---|---|---|
| Level or log φ for Test 1 | Level stays the gate; log is reported beside it | Switching after seeing that log passes everything would mean picking the rule that passes |
| BAB, which cannot be traded | Stays in the paper track; primary results are labelled an **upper bound** | Keeping it is the rule written before results; dropping it now would follow the results |
| Leverage cap | 1.5× in the primary test; 1.0×, 2.0× and none in Phase 8. The ETF track keeps its short SPY leg | Set in the research plan, and first written into code on 2026-10-17. Uncapped sizes reach 11.6× |

**Outcome.** The primary passes (φ 0.514; α 4.2% a year, t 6.1), as an upper bound. Equal weight, shrunk mean-variance and every market + momentum mix fail Test 1 on level φ. The ETF track has 48 months and is not judged.

### 6. Fixed for the phases still to come

None of the results these rules govern exist yet.

- **Primary specification.**
  - Assets: the paper track (market, momentum, BAB).
  - Weights: inverse volatility, with the recipe set at each year-end from past data only.
  - Order: combine first, then time.
  - Size: capped at 1.5×.
  - Variance forecast: the Phase 6 choice, made on the development sample only.
- **Benchmark.** The buy-and-hold mix with the same recipe.
- **Upper bound.** Every result of the primary is labelled an upper bound, because paper BAB cannot be traded.
- **Phase 8, robustness.** Caps of 1.0×, 2.0× and none, and timing each piece separately before combining. These checks are reported beside the primary and never replace it.
- **Phase 9, the final test.** It runs once, on the holdout (2019-01-01 onward), after every item in Section 7 has been fixed in a dated entry. The holdout is opened only through `load_holdout(confirm="phase 9 final test")`. Nothing seen in the holdout informs any model choice.

### 7. Still open: to be fixed in a dated entry before the holdout is opened

1. **The success threshold.** It must say what counts as "beats buy-and-hold": the statistic (for example, the difference in Sharpe ratios), its test and its cutoff. With the 1.5× cap, the timed mix runs below buy-and-hold's volatility (5.3–5.6% against 7.0–7.2% on the development sample), so the comparison has to be scale-free. It is best fixed before Phase 7, whose walk-forward will show the size of the development-sample gain.
2. **Costs.** The cost model, including spreads (unverified since Phase 3) and the cost of the ETF track's short SPY leg.
3. **Phase 6.** The out-of-sample criterion for choosing the variance model. The candidates are listed in Phase 2's closing table.
4. **Phase 7, the walk-forward.** Four questions are carried over from Phase 3:
   - what to hold when no asset passes the screen;
   - how to treat assets that flip in and out;
   - whether to evaluate on the long paper history, the ETF era from 2012, or both;
   - whether filter 1 uses level or log φ for momentum. The Phase 4 call on Test 1 covers p, not the screen.
5. **The ETF track in the final test.** Whether it is reported on the holdout as the tradeable secondary result.

---

## Entry 2 — 2026-10-18: Phase 5, the backtest engine and the cost model

**Status.** These rules come from the research plan's Phase 5 and were written into `src/engine/` on 2026-10-18. They were fixed **before any real-data result under them**. The engine was verified on simulated data only. Its single real-data run (the Phase 4 primary book) checked dates, bounds and look-ahead, and displayed no return statistic. Phase 4 has already shown real-time timed Sharpe ratios for the primary under Phase 4's own *c* (1.40 capped, 1.23 uncapped, before costs), so these rules are not blind to that result.

This entry fixes Entry 1, Section 7, item 2 (costs), except that spreads remain unverified: κ stands in for them, and the break-even κ is the safety margin.

### 1. Timing rules (`src/engine/__init__.py`: `TIMING`, `ENGINE_RULES`)

| Rule | Size | Role |
|---|---|---|
| Inverse variance | c_t / σ̂²_t | **primary** (Moreira and Muir) |
| Inverse volatility | c_t / σ̂_t | reported (Barroso and Santa-Clara) |
| Downside inverse volatility | c_t / σ̂⁻_t, σ̂⁻ = √RV⁻ | reported (Wang and Yan) |

Until Phase 6 chooses a model, σ̂²_t is the month's realized variance RV_t, and RV⁻_t is Phase 2's downside RV. A fixed 12% annual target (c = 0.12/√12 with σ̂ from 126 days) is used only to replicate Barroso and Santa-Clara on momentum.

### 2. c_t in real time (`weights.realtime_c`)

c_t = sd(r_{s+1}) / sd(r_{s+1} × signal_s), over every pair with s+1 ≤ t (an expanding window), and only once 120 pairs exist. Signal and returns come from the mix that will be held next month, applied to all past data, so January's size uses December under the new recipe.

**Replaces Phase 4's c**, which matched variances over each recipe's 10-year window and was set once a year.

### 3. Execution (`execution.py`)

- The size decided at the close of month t's last trading day is traded at that close and held through month t+1. `delay` = 0 in the primary test.
- 0 ≤ size ≤ 1.5 in the primary test. Phase 8 tries 1.0 and 2.0 (Entry 1 also lists no cap).
- Positions drift within the month: w̃_i = w_i (1 + R_i) / (1 + R_p). The account starts in cash.
- Monthly returns are exact excess returns: compounded total return minus compounded risk-free return. This replaces Phase 4's daily-rebalanced constant exposure.
- Zero-cost factors are treated as assets whose total return is rf plus their excess return.

### 4. Costs (`execution.COSTS`)

| Cost | Rule |
|---|---|
| Trading | κ × \|w_new − w̃\| per instrument, paid when the new position goes on: κ = 5 bp for SPY, 10 bp for MTUM, QUAL and USMV, one-way per unit of weight |
| Borrowing | cash below zero pays rf + 0.5% a year, accrued daily |
| Short fee | 0.25% a year on the SPY short behind the MTUM tilt, accrued daily. The plan says "a small fee"; 0.25% is a choice, not a quote |
| Paper track | before costs (κ = 0, no spread or fee); every paper result is an upper bound |
| Break-even | reported two ways: the one-way κ at which the timed Sharpe ratio falls to buy-and-hold's (the plan's), and the κ at which each strategy's mean excess return is zero. A round trip is 2κ |

### 5. Benchmark

The same book at a constant size of 1, traded on the same dates and paying the same costs.

### 6. Placebo rule (applied to real data from Phase 6)

Spanning α of the timed excess return on buy-and-hold's. The forecast is reordered at random in time 1,000 times, and weights (c_t included) and execution are rerun each time. **The real α must beat at least 95% of the shuffled α's.**

### 7. Engine checks: outcome on simulated data (`test/test_engine_checks.py`, `reports/phase5_engine.ipynb`)

The checks and their expected outcomes are the plan's. Three details were set after pilot runs on simulated data:

- **One 500-year history per world**, for power: 90 years give t > 1.96 in only 70% of world-A histories.
- **The unpredictable-variance world, and its threshold of a 0.1 Sharpe jump, for the oracle.** In the GARCH worlds, last month's RV leaves the oracle little to add. Across ten pilot seeds the capped jump was 0.11–0.19, so 0.1 has little slack.
- **"Centered" as a mean within 0.1 standard deviations of zero**, set after 3,000 draws showed a bias of 0.04 standard deviations from the real-time c_t and the cap. With a fixed c and no cap, where theory gives exactly zero, the bias is not significant (z 1.2).

| Check | Pass condition | Outcome |
|---|---|---|
| Two worlds (GARCH, 500 years, seed 0) | A (μ constant): α > 0, t > 1.96. B (μ ∝ σ²): t < 1.96 | A: α +1.95%, t 6.0. B: t 0.4. **Pass** |
| Constant forecast | size 1 with real-time c_t; with a fixed c, timed = size × buy-and-hold, α ≈ 0, R² = 1 | gaps of order 10⁻¹⁶. **Pass** |
| Shuffled forecast (1,000) | mean within 0.1 standard deviations of 0; placebo passes in A, fails in B | A: real beats 100%. B: real beats 67%. **Pass** |
| Oracle | A: buy-and-hold < baseline < oracle. Unpredictable variance: baseline t < 1.96, oracle Sharpe ≥ buy-and-hold + 0.1 | A: 0.39 < 0.47 < 0.50. Unpredictable: t 1.2, +0.18. **Pass** |
| Hand check (`reports/phase5_hand_check.xlsx`) | 70 numbers at 5 months match a spreadsheet to 10⁻¹¹ | largest gap 2 × 10⁻¹⁴; seven planted bugs each caught. **Pass** |

### 8. Added to Entry 1's Section 7: to fix before the holdout opens

6. **The ETF track's c_t.** Under the 120-pair rule the ETF mix (MTUM from 2013) has no size before 2023. Choose a shorter minimum, a c_t taken from the paper market + momentum twin, or an untimed ETF report.
7. **The real-data engine check.** Replicate Moreira and Muir (market) and Barroso and Santa-Clara (momentum, 12% target) on the development sample, as Phase 6's first step. Record which break-even definition Barroso and Santa-Clara use.
8. **Power.** At about 90 years, a 5% test on alpha detects a true edge of world A's size only about 70% of the time. Item 1, the success threshold, should account for this.

---

## Entry 3 — 2026-10-18: decisions on Phase 5's open items, and the success threshold

**Status.** Written after Phase 5's simulated checks and before any real-data result under these rules. Two things were already known when it was written:

- Phase 4's development-sample numbers for the primary: timed Sharpe 1.40 against buy-and-hold's 1.17, before costs, under Phase 4's c and daily rebalancing.
- The power results of Phase 5's simulations (Section 5 below).

**Resolves** Entry 1, Section 7, item 1 (the success threshold), and Entry 2, Section 8, items 6–8. Still open from Entry 1, Section 7: item 3 (Phase 6's model-selection criterion), item 4 (Phase 7's four questions) and item 5 (the ETF track's role in the final test).

### 1. Phase 5's choices are confirmed, with three changes

All seven choices of the Phase 5 report stay. Three of them change:

- **Short fee.** 0.25% a year on the SPY short is the base case. Phase 8's cost row adds a stress test at 1% a year, to cover a low or missing short rebate (`execution.COSTS["etf_short_stress"]`).
- **Volatility in every report.** The timed run and the benchmark both report realized volatility beside every comparison. So a gap in certainty-equivalent return cannot hide a gap in size.
- **No comparison with Barroso and Santa-Clara's cost figures.** Both kinds of break-even are reported, but neither is compared with Barroso and Santa-Clara's 1.27 and 1.75 points. Those figures are the costs that make momentum's profits insignificant at 5%, and they appear to include the trading inside the factor. This replaces Entry 2, Section 8, item 7's "record which break-even definition Barroso and Santa-Clara use".

### 2. Phase 6 opens with a replication

Phase 6 first replicates the published single-factor results only: momentum's Sharpe ratio rising from 0.53 to 0.97 over 1927–2011 (Barroso and Santa-Clara). The average size w_t is reported, so the costs inside each factor can be judged.

### 3. The ETF track's c_t needs 36 months

This replaces the 120-month minimum for the ETF track only (`ENGINE_RULES["etf_min_months"]`), matching Phase 7's three-year rule. Why it is safe, and why the alternatives were rejected:

- c_t only sets the average size, and the Sharpe ratio ignores size, so a shorter history changes the scale, not the test.
- Borrowing the paper twin's c_t would give the wrong scale.
- Waiting until 2023 would waste most of the holdout.

On the development sample the ETF book is timed from June 2016 (31 months).

### 4. The primary test and its readings

| Item | Rule |
|---|---|
| Statistic | Sharpe gain: Sharpe(timed) − Sharpe(buy-and-hold), a year, from the run's monthly excess returns (after costs where the track has costs) |
| Test | One-sided at 5%; H0: gain ≤ 0. The threshold stays at 5%: loosening it would buy power with false positives |
| Standard error | The delta method of Ledoit and Wolf (2008), with a Newey-West covariance and 6 lags (`evaluation.sharpe_gain_test`). This method is a choice made with this entry; the decision fixed the test's side and level |
| Readings | In this order: **edge** if the test rejects; **no meaningful edge** if the one-sided 95% upper bound of the gain is below 0.10 a year; **inconclusive** otherwise |
| Reported with it | The minimum detectable effect at 80% power, (z₀.₉₅ + z₀.₈₀) × standard error; the realized volatility of both runs; the average size |
| Applies to | The primary specification of Entry 1, Section 6, in the Phase 9 final test |

### 5. Power, re-run on the primary test (simulated)

This replaces Entry 2, Section 8, item 8, which measured power on alpha. Source: `reports/phase5_engine.ipynb`, Section 4. Inverse-variance timing, 1.5× cap, costs off.

| Sample | World | Test rejects | Minimum detectable effect (80% power) | Readings: edge / inconclusive / no meaningful edge |
|---|---|---|---|---|
| 90 years (100 histories) | A: μ constant (true gain about 0.09) | 30% (alpha's t-test: 70%) | 0.18 | 30% / 61% / 9% |
| 90 years | B: μ ∝ σ² (no gain) | 1% | 0.17 | 1% / 15% / 84% |
| 92 months, the holdout's length (200 windows) | A | 10% | 0.46 | 10% / 83% / 7% |
| 92 months | B | 4% | 0.46 | 4% / 73% / 24% |

The volatility rules have smaller minimum detectable effects over 90 years (0.11–0.12) than inverse variance. On holdout-length data the final test can call an edge only if the gain is several times world A's, and it can call "no meaningful edge" only if timing loses by about 0.2 a year or more. These are reported beside the result. They do not change the rule.

---

## Entry 4 — 2026-10-18: Phase 6, the forecast models and the selection rule

**Status.** Written before any real-data forecast was made or scored, and before Phase 2's hints were re-run on the primary series p. What was known when it was written:

- Phase 2's tests on the single factors (long memory everywhere; VIX helps market, value, momentum and BAB; bad days raise risk only for value).
- Phase 4's development-sample numbers for the primary, under Phase 4's c and the last month's RV.
- Phase 5's simulated results.

The models were built and tested on simulated data only (`test/test_models.py`). Two checks against the `arch` package also ran on simulated data: GARCH and GJR estimates agree to four decimals, and the MCS agrees on membership.

**Resolves** Entry 1, Section 7, item 3 (Phase 6's model-selection criterion). The selection rule and the scorecards are the research plan's. Every rule marked *(my choice)* was set with this entry by Claude, not by the plan.

### 1. What every model forecasts (`src/models/inputs.py`)

| Item | Rule |
|---|---|
| Target | At the close of each month's last trading day (the origin): Σ r² over the next 22 trading days, for the mix held next month. 22 days is one holding month. A weekly rule would use 5 |
| Data | Data up to the origin's close. Every model makes its first forecast at the end of the mix's 120th month, so all eight are scored on the same months *(my choice: the project's 10-year minimum)* |
| Window | Expanding |
| Refit | Every month-end for M2–M6. Yearly for M7, at its first forecast and every December, because boosting with an inner walk-forward per recipe is the costly model. GARCH is cheap enough to refit monthly |
| Embargo | A direct model's training row at day d has the target over d+1..d+22, so a refit at origin R uses only rows with d ≤ R − 22 |
| Tuning | M7 only, by an inner walk-forward inside the training rows (Section 2) |
| Per recipe | The engine forecasts the mix held next month over all its history (Entry 2). So every model runs its own walk-forward on each recipe's mix, each forecast made only from data up to its origin |

### 2. The eight models (`src/models/`)

| Model | Definition |
|---|---|
| M0 Rolling RV | Σ r² over the last 22 trading days |
| M1 EWMA | σ̂²_{d+1} = 0.94 σ̂²_d + 0.06 r²_d, started at the first r²; forecast 22 σ̂² |
| M2 GARCH(1,1) | Gaussian quasi-maximum likelihood on daily returns in percent, **zero mean** *(my choice: ε = r, the same r² as RV)*. Recursion started at the mean r² of the estimation data. Constraints: ω > 0, a, b ≥ 0, a + b < 1. Forecast: Σ_{h=1..22} [σ̄² + (a+b)^{h−1}(σ²_{t+1} − σ̄²)] |
| M3 GJR-GARCH | M2 plus g ε²_{d−1} 1{ε_{d−1} < 0}, with a + g ≥ 0 and a + b + g/2 < 1. The forecast decays at rate a + b + g/2 |
| M4 HAR, log | Ordinary least squares, on every trading day, of log(22-day forward RV) on d, w, m. These are the logs of the mean r² over the last 1, 5 and 22 days. The 1- and 5-day means are floored at 1/1000 of the 22-day mean before the log *(my choice: a near-zero return has log r² near −∞, down to −18 below its mean; the floor binds on about 2.6% of days, as normal theory predicts)*. Forecast exp(fit + s²/2), with s² the residual variance |
| M5 Downside HAR | M4 plus the down-day share of each lag: Σ r² on down days / Σ r² over 1, 5 and 22 days *(my choice: in logs, a split into log downside and log upside variance fails, because one of the two is zero every day; the shares carry the same split)* |
| M6 HAR-X | M5 if Phase 2's Test 3 passes on the series forecast (δ > 0 with t > 1.96), else M4, plus two inputs: log((VIX/100)²/12), and the counts of scheduled FOMC, CPI and jobs-report days in the next 22 trading days (scheduled in advance, so known at the origin). The extra inputs enter a refit once they have 60 months of history *(my choice)*: VIX from about 1995, events from about 1999. The regression is then fitted on the rows that have them, and before that M6 equals its base model |
| M7 Gradient boosting | Inputs: M6's. Loss: gamma deviance, which is 2 × QLIKE, fitted on the 22-day RV with a log link (so no back-transform is needed) *(my choice)*. Rows: every fifth trading day *(my choice, for compute: neighbouring targets overlap by 21 days)*. Inner walk-forward at each refit: each of the last three years of training rows is forecast by a model trained on rows whose targets end before that year starts, and the setting with the lowest mean QLIKE wins. Grid *(my choice)*: 4, 8 or 16 leaves × 25, 50, 100, 200 or 400 trees; learning rate 0.05; at least 50 rows a leaf; no early stopping (its random split would leak across overlapping targets). Missing VIX and event values stay missing |

**Simplicity order** (the selection rule's): M0, M1, M2, M3, M4, M5, M6, M7.

### 3. Scorecard A: forecast accuracy (`src/models/scorecard.py`)

| Item | Rule |
|---|---|
| Loss | QLIKE_t = RV_t / σ̂²_t − ln(RV_t / σ̂²_t) − 1, with RV_t the 22-day realized variance after the decision, for the mix held. This is the forecast the engine actually uses each month |
| Months | Every holding month of Scorecard B whose 22-day target lies inside the development sample (all but the last) |
| Reported | Mean QLIKE (the score), median QLIKE, mean RV / mean σ̂² (calibration) |
| Model Confidence Set | Hansen, Lunde and Nason (2011), T_max statistic, eliminated down to one model with MCS p-values. The set is the models with MCS p-value ≥ 10% (a 90% set). Stationary bootstrap with a mean block of 12 months, 10,000 draws, seed 0 *(block, draws and seed: my choice)* |

### 4. Scorecard B: strategy value

| Item | Rule |
|---|---|
| Specification | The primary book (Entry 1, Section 6), with only the forecast changing. Inverse-variance timing, inverse-volatility mix, monthly, cap 1.5, real-time c_t from 120 pairs. Costs are Entry 2's for the track, which is none for the paper factors. Turnover and the break-even κ are reported, so a model that trades more shows |
| Months | The development walk-forward, the same months for all eight: from the first month with a c_t to December 2018 |
| Reported | Sharpe ratio; CER = 12 × (mean − γ/2 × variance) with **γ = 5** *(my choice)*; realized volatility (Entry 3); mean; the primary test's Sharpe gain over buy-and-hold with its reading; spanning α; mean size; share of months capped; turnover; break-even κ; the engine's placebo (1,000 shuffled forecasts, pass at 95%) |
| Reference rows (not candidates) | Buy-and-hold, and the oracle: the holding month's actual RV, which looks ahead. The oracle-minus-M0 gap is the room any forecast has to win |

### 5. The selection rule (the plan's, fixed here)

**Pick the simplest model inside the MCS whose Sharpe ratio is within one standard error of the best.** Then freeze it; its clean test is the holdout.

- **Best**: the highest Sharpe ratio of the eight on Scorecard B.
- **One standard error** *(my choice of definition)*: the standard error of the gap Sharpe(best) − Sharpe(model), by the primary test's method (Ledoit-Wolf delta method, Newey-West 6 lags, `evaluation.sharpe_difference`). A model qualifies if its gap is at most that error. The eight runs share their months and most of their returns, so the paired error says whether two models tie. The unpaired error of the best model's own Sharpe ratio would be about 0.15 over 68 years, and every model would qualify.
- **Fallback** *(my choice)*: if no model inside the MCS qualifies, the MCS condition is dropped, and the pick is the simplest model within one standard error of the best. The best model itself always qualifies.
- **Q4**: the Spearman correlation between the QLIKE ranking and the Sharpe ranking of the eight. Near zero means forecast accuracy is not the bottleneck. With eight models it is reported, not tested.

### 6. Two frozen choices

| Choice | Chosen on | Applied to |
|---|---|---|
| Combined portfolio | the primary book (paper track, inverse-volatility mix) | the primary specification from Phase 7 on |
| SPY reference *(my reading of the plan)* | the market factor, as SPY's long-history stand-in, timed alone. Costs are SPY's (5 bp a trade, borrowed cash at rf + 0.5%). ETFs are never tuned on (Entry 1, Section 2) | SPY timed alone, from Phase 7 on |

M6's base (M4 or M5) is set separately for each, by Test 3 on that book's series: p for the primary, the market factor for the SPY reference.

### 7. Order of work, and what is information only

1. The replication of Entry 3, Section 2: Barroso and Santa-Clara's momentum (0.53 to 0.97, 1927–2011: 126-day variance, 12% target, no cap) and Moreira and Muir's market (α about 4.9% a year, 1926–2015: full-sample c, no cap). Average size reported. Our momentum is French's UMD factor, not their decile spread, so its numbers should be close, not equal.
2. Phase 2's Tests 2, 3, 5 and 9 on each book's series. They set expectations and M6's base, nothing else.
3. Both scorecards for all eight models, then the rule.

Information only, never changing the choice:
- the VIX era (from the first month M6 uses VIX);
- the oracle;
- the placebo;
- the single-factor replications.

---

## Entry 5 — 2026-10-18: Phase 6 outcome, one fix after the first run, and the frozen choices

**Status.** Written after the Phase 6 runs. It records three things:

- one change made after the first run: a fix to GARCH estimation;
- the outcome under Entry 4's rules;
- the two frozen choices.

Entry 4 is not edited. Every Entry 4 rule stays; the fix changes how M2 and M3 are estimated, not what they are or how they are scored.

### 1. The first run, and why its choice is void

**How it was found.** The first run's Scorecard A gave M3 (GJR-GARCH) on the primary book a mean QLIKE of 0.957, against 0.30–0.39 for the other seven. Its mean RV / mean forecast was 0.0000. Its forecasts reached 5.7 × 10¹⁰ times M0's (recipe 54, held in 1995).

**The cause.** In some refits the SLSQP optimizer reported success at a point far worse than its start, for example log-likelihood −5,865 against −16,796 at its warm start, in recipe 0's February 2007 refit of M3. ω, the GARCH intercept, came out about 10⁹ times too large and b at 0. Each refit starts from the last estimate, so the bad point was carried forward month after month. On the primary book, 1,195 of 73,164 M3 refits and 105 of the M2 refits were degenerate (ω above the mean r², or b below 0.5). The SPY reference had none. The unscaled problem was the root: ω is about 1/1,000 of the other parameters, and SLSQP is sensitive to scale.

**Why it touched the choice.** The first run chose **M0** for the primary book, but only because M0 sat inside the MCS, at p = 0.117. That p-value was inherited from the elimination round that removed M3, whose wild losses made the round's test too noisy to reject. It chose **M2** for the SPY reference.

**The fix** (`filters.garch_fit`, 2026-10-18):
- ω is optimized as a share of the mean r², so every parameter is of order 0.01–1, and ω is capped at the mean r².
- A refit's result is kept only if its likelihood is at least as good as its start's.

Nothing else changed: the models, the data, the refit schedule, both scorecards and the selection rule are as in Entry 4.

**Checks of the fix.**
- On real refits of two recipes (1940, 1979 and 2018; GARCH and GJR), the estimates match the `arch` package to four decimals. No degenerate refit remains.
- A new test (`test_a_garch_refit_never_ends_worse_than_its_start_and_keeps_omega_in_scale`) fails on the old code and passes on the new.
- The build now stops if any forecast is more than 1,000 times, or less than 1/1,000 of, M0's for the same month (`build.sanity`). Real forecasts stay within about 10 times of each other.
- In the 2018 recipe, 11–17% of refits still stop short of the optimizer's tolerance. They sit at the optimum of the month before and never at a worse likelihood than it.

**Not changed by the fix.** HAR (M4–M6) and boosting (M7) were not recomputed: the cache now keys each model on its own code. Their cached forecasts equal a fresh computation to the last digit, at three primary recipes and the SPY reference. The SPY reference's Sharpe ratios move by at most 0.0005, and its choice is unchanged. The first run's tables are kept in `reports/phase6_first_run_*.csv`.

**What was seen first.** The fix was made after the first run's scorecards and choice were seen. It was made because no reading of the plan allows a forecast of 10¹⁰ times the realized variance, not to change the choice. The re-run decides, whatever it chooses.

### 2. Outcome (development sample; `reports/phase6_scorecard_*.csv`, `reports/phase6_selection.csv`, `reports/phase6_forecasts.ipynb`)

**Samples.**
- Primary book: 817 holding months, December 1950 to December 2018; Scorecard A uses 816.
- SPY reference: 870 holding months, July 1946 to December 2018; Scorecard A uses 869.

**M6's base.** By Test 3, M4 for both books. Test 3 failed on both: δ = −0.03 (t −0.5) on p, and +0.06 (t +1.1) on the market.

| Book | Lowest QLIKE | Outside the MCS | Highest Sharpe | Sharpe range of the eight | Buy-and-hold Sharpe | Oracle Sharpe | Rank correlation (Q4) |
|---|---|---|---|---|---|---|---|
| Primary | M6 (0.3025) | M0 (p 0.001) | M3 (1.567) | 1.509–1.567 | 1.211 | 2.222 | +0.10 |
| SPY reference | M6 (0.2727) | M0, M1, M7 | M3 (0.533) | 0.503–0.533 | 0.497 | 1.124 | +0.62 |

**The selection.**

- **Primary: M3, GJR-GARCH.** It has the best Sharpe ratio, and no simpler model inside the MCS is within one standard error of it. M0 is within one error but outside the MCS. M1 misses by 0.002 (gap 0.042, error 0.040), and M2 by 0.005 (gap 0.015, error 0.010). The fallback was not needed. **This is a knife-edge choice.** Had M1 qualified, the rule would have picked EWMA.
- **SPY reference: M2, GARCH(1,1).** It is the simplest model inside the MCS within one standard error of the best (M3; gap 0.006, error 0.014). M0 and M1 are within one error but outside the MCS.

**Readings, all information only.**
- **Primary book.** Every model reads "edge" on the primary test: Sharpe gains of 0.30–0.36 over buy-and-hold, p ≤ 0.004. All pass the placebo, beating 100% of shuffled alphas. These are development-sample, before-cost numbers on a book holding untradeable BAB, so they are an upper bound. They are not the final test.
- **SPY reference.** Every model reads "inconclusive": gains of 0.006–0.036. Seven of eight pass the placebo; M7 fails at 94.5%.
- **The oracle.** The gap between the oracle and M0 (0.71 primary, 0.61 SPY) is ten to twenty times the spread among the eight models (0.06, 0.03). So the room to win is large, and no real forecast here takes much of it.
- **Q4.** For the primary book, accuracy barely ranks strategy value (+0.10). The GARCH models win on Sharpe while sitting mid-table on QLIKE. Their smoother forecasts let real-time c_t hold a larger average size (1.00 against M0's 0.67), and the 1.5 cap then binds in more months (25% against 11%).
- **CER.** CER follows size at these low volatilities, as Entry 3 warned. M0 and M1 beat buy-and-hold on Sharpe but trail it on CER.

### 3. Frozen choices

| Choice | Model | Specification |
|---|---|---|
| Combined portfolio | **M3 GJR-GARCH** | zero-mean Gaussian QMLE on daily returns of the mix held next month; expanding window; refit every month-end with this entry's safeguards; the 22-day forecast is the sum of h-step forecasts decaying at a + b + g/2 |
| SPY reference | **M2 GARCH(1,1)** | the same, with g = 0 |

Both are frozen from Phase 7 on. Their clean test is the holdout. The choice for the combined portfolio rests on a 0.002 margin in the one-standard-error rule. That is reported beside every result that uses it, and it is not revisited.

---

## Entry 6 — 2026-10-19: Phase 7, the walk-forward and its tests

**Status.** Written before any walk-forward result. Four things were known when it was written:

- Phase 6's results, on the full-sample asset list (Entry 5).
- The real-time screen at every year-end (Phase 3, Figure 6), re-tabulated for this entry. It shows which factors pass, not any return.
- A mechanics-only run of the walk-forward on the real panel. It printed dates, counts, segments and the range of sizes, and no return statistic, as Entry 2's engine check did.
- A rough power figure: Phase 6's standard error of the Sharpe gain (0.10 over 817 months), scaled to about 276 months, gives a minimum detectable effect of about 0.43.

The user made the four calls in Section 1 on 2026-10-19, after seeing the screen's admissions and that power figure, and before any return under them. Every rule marked *(my choice)* was set with this entry by Claude, not by the plan or the user.

**Resolves** Entry 1, Section 7, item 4 (Phase 7's four questions) and item 5 (the ETF track's role). With this entry, every item that Entries 1 and 2 left open is fixed.

### 1. The four calls (the user's, 2026-10-19)

| Question | Call | What it means here |
|---|---|---|
| Which assets the primary book holds | The real-time screen over all seven factors, at every year-end, on filters 1, 3 and 6. Filters 4 and 5 judge an ETF, and no ETF exists before 1993 (SPY) or 2011–2013 (the factor ETFs), so the paper track does not apply them. Its results remain an upper bound | Market in 1984 and 1997–2018; BAB in 1997–2001 and 2003–2018; profitability in 2001–2013 and 2016; momentum in 2010–2018. Phase 6's fixed book is reported beside it as information |
| Level or log φ in filter 1 | Level φ, as Phases 2 and 3 wrote it, consistent with the Phase 4 call on p | Momentum stays out until its level φ crosses 0.5 at the end of 2009 |
| What to hold when nothing passes, or before c_t exists | Cash, in the timed run and buy-and-hold alike; those periods are not scored | 1985–1996 are in cash. So is 1937, momentum's one-year accident in Phase 3, which has no c_t yet |
| Assets that flip in and out | No rule: hold what passes at each year-end | Profitability is held in 2001–2013, drops out for 2014–2015, and returns for 2016; BAB skips 2002 |

### 2. The walk-forward (`src/walkforward/`)

| Step | Rule |
|---|---|
| Window | Expanding everywhere. The first 10 years of each series are initial training, as in Cederburg et al.: the screen needs 120 months of a factor, the models first forecast at the end of the mix's 120th month (Entry 4), and c_t needs 120 monthly pairs after that (Entry 2) |
| Screen | `screen(panel, ohlc, asof)` at every year-end, reading data up to that date only. Survivors pass filters 1, 3 and 6 and have 120 months of history |
| Recipe | Phase 4's Steps 1 and 2 on the survivors, at the same year-end: the duplicate check (ρ > +0.7 on all or crisis months, judged on 120 common months), then each of the four mixes from the last 10 years of daily returns. On the paper track a position is its mix weight |
| Holding | A recipe is held from the first rebalance on or after its year-end until the first rebalance on or after the next one. A monthly book holds it over the next calendar year |
| Forecast | At each rebalance close, the variance of the mix held next period, from data up to that close: Entry 4's models and refit schedule |
| c_t and size | Entry 2: real-time and expanding, on the mix held next; size = c_t × signal, capped at 1.5 |
| Trade | At the decision close (Entry 2: delay 0), so the position earns from the next trading day. This is the plan's one-day lag between the signal and the return it earns. A later trade is Phase 8's |
| Cash | A period after a year-end with no survivors, or before its recipe's c_t exists, holds cash in both runs and is not scored. The account re-enters from cash, so the first trade of each stretch pays for the whole entry |
| Scored months | 276, from the mechanics run: January–December 1984, and January 1997 to December 2018 |

### 3. The primary test (it stands alone)

| Item | Rule |
|---|---|
| Specification | The screened paper book; inverse-volatility mix; M3 GJR-GARCH (Entry 5); inverse-variance timing; monthly; cap 1.5; real-time c_t from 120 pairs; no costs (paper factors) |
| Months | Every month it scores |
| Statistic, test, readings | Entry 3's: the Sharpe gain over buy-and-hold, one-sided at 5%, by the Ledoit-Wolf delta method with a Newey-West covariance (6 lags), read as edge, no meaningful edge or inconclusive. It was fixed before the data, so its p-value is read as is |
| Reported with it | Entry 3's minimum detectable effect, both volatilities and the mean size; the metrics of Section 5; the CER and combination tests of Section 6; the placebo (information) |
| Labels | An upper bound: paper BAB and profitability cannot be traded, and there are no costs. M3 was chosen by a 0.002 margin (Entry 5) |
| What it is not | The final test. That is Phase 9's, on the holdout (Entry 1, Section 6) |

### 4. The family of 192 variants

8 forecasts × 3 timing rules × 4 mixes × 2 rebalancing frequencies. Everything else is as in the primary, which is one of the 192.

| Dimension | Values |
|---|---|
| Forecast | M0–M7 (Entry 4). M6 and M7 build on M4, by Phase 6's Test 3 on the combined portfolio *(my choice: not re-tested on the screened book)* |
| Timing rule | inverse variance (power 1); inverse volatility (power 0.5); downside inverse volatility (power 0.5 on σ̂⁻²) |
| Mix | equal; inverse variance; inverse volatility; shrunk mean-variance (Phase 4) |
| Rebalancing | monthly; weekly |

**The downside rule under each model** *(my choice)*. σ̂⁻² = σ̂² × s, where s is the share of Σ r² over the last 22 trading days that comes from down days. If those 22 days hold fewer than three down days, s uses the last 44, which is Phase 2's rule for downside RV (Wang and Yan). Under monthly M0, σ̂⁻² is then the downside RV of the last 22 days. Without the extension, a stretch of 22 up days would give a forecast of zero, and the mechanics run found such stretches in the mixes' history. Phase 5's calendar-month downside RV is not used.

**Weekly variants** *(my choice of the details; the plan names only the frequency)*.
- They rebalance at the close of each week's last trading day (weeks end on Friday).
- Every model forecasts the next 5 trading days' variance (Entry 4, Section 1: "a weekly rule would use 5"):
  - M0: Σ r² over the last 5 days. M1: 5 × the EWMA variance.
  - M2 and M3: the same month-end fits as the monthly models, with σ² filtered up to the weekly origin and 5 h-step forecasts summed.
  - M4–M6: refit at each month-end on the 5-day target, with a 5-day embargo; M6's event counts cover the next 5 days.
  - M7: refit at its first forecast and each December on the 5-day target, rows every fifth day, inner walk-forward with a 5-day embargo.
  - Each weekly origin uses the latest month-end fit on or before it. The first weekly forecast is at the first week ending on or after the first monthly one.
- c_t needs 520 weekly pairs (ten years of 52 weeks).
- A recipe takes over at the first weekly rebalance on or after its year-end.
- A weekly run is scored on calendar months. Each week goes to the month of its closing trade, and a month's excess return is its compounded net return minus its compounded risk-free return. So all 192 variants are measured on monthly returns, with the same annualization and Newey-West lags.

**The family's months** *(my choice)*: the months every variant scores. These are the primary's 276; a weekly run's partial January 1985 drops out.

### 5. Metrics (calendar-month excess returns, for the timed run and buy-and-hold)

| Metric | Definition |
|---|---|
| Sharpe ratio | mean / standard deviation × √12 |
| CER | 12 × (mean − γ/2 × variance), γ = 5 |
| Sortino ratio | mean / √mean(min(r, 0)²) × √12. The target is 0 because r is already in excess of the risk-free rate |
| Maximum drawdown | the largest fall of Π(1 + r) from its running peak |
| Turnover | Σᵢ \|wᵢ − w̃ᵢ\| per rebalance, its mean and its sum a year, leaving out each stretch's entry trade |
| Break-even cost | the one-way κ at which the Sharpe gain is zero (Entry 2's first kind) |
| Also | realized volatility (Entry 3), mean, spanning α, mean size, share of periods capped |

### 6. Tests

| Test | Rule |
|---|---|
| Sharpe gain | Entry 3's: Ledoit and Wolf (2008), delta method with a Newey-West covariance (6 lags), one-sided at 5%. Not Jobson-Korkie, which assumes normal, independent returns |
| CER gain | CER(timed) − CER(buy-and-hold), one-sided. p = the share of bootstrap draws with ΔCER* − ΔCER ≥ ΔCER *(this centered form: my choice)* |
| Real-time combination (Cederburg et al. 2020) | Each month after the first 120 scored months, a γ = 5 investor splits the account among buy-and-hold, the timed run and cash, with w = Σ̂⁻¹μ̂ / γ estimated on the scored months before it (expanding; sample mean and covariance). If the absolute weights sum to more than 5, w is scaled down to 5. The benchmark investor holds buy-and-hold and cash, w = μ̂ / (γ σ̂²), clipped to ±5. The statistic is the difference of their CERs (γ = 5) over the out-of-sample months, one-sided, with the CER test's bootstrap p-value. Out of sample means January 2006 to December 2018, 156 months. The combination's own trading is not charged *(my choice; the paper book has no costs anyway)* |
| Bootstrap | Stationary (Politis and Romano 1994), mean blocks of 6 and 12 months, 10,000 draws each, seed 0. **Every bootstrap p-value is the larger of the two**, so the block length cannot help a result *(my choice of how to use the plan's "6 to 12 months")*. All variants are resampled with the same draws, which keeps their correlation |

### 7. Multiple testing, Q3 and Q4

| Item | Rule |
|---|---|
| The primary | Stands alone; its p-value is read as is (Section 3) |
| Romano-Wolf | Stepdown over the 192 Sharpe gains over buy-and-hold, one-sided, familywise error 5%. Each statistic is gain / se, with se its Ledoit-Wolf standard error. The bootstrap statistics are (gain* − gain) / se, with se from the real sample *(my choice: this fixed studentization, as in Hansen's SPA test, keeps 10,000 draws of 192 variants fast)*. A variant beats buy-and-hold after adjustment if its adjusted p is below 0.05 |
| Holm-Bonferroni | On the 192 raw Ledoit-Wolf p-values, as the conservative cross-check. The smallest must be below 0.05 / 192 ≈ 0.00026 |
| Deflated Sharpe ratio | Bailey and López de Prado (2014), for the variant with the largest Sharpe gain *(my choice of "best")*, measured against buy-and-hold *(my choice)*. The threshold is that variant's own buy-and-hold Sharpe ratio plus √V × E[max of N standard normals], with N = 192 and V the variance of the 192 monthly Sharpe gains. DSR = Φ((SR − threshold) √(T − 1) / √(1 − skew × SR + (kurtosis − 1)/4 × SR²)), in monthly units; it passes at 0.95. Against zero, as in the paper, it would only say that a mix whose buy-and-hold Sharpe ratio is near 1 beats nothing |
| Q3: does a better forecast beat the baseline? | In each of the 24 cells (rule × mix × frequency): Sharpe(model) − Sharpe(M0), by Ledoit-Wolf, one-sided (the model beats M0), with Romano-Wolf over the 168 comparisons. Read beside it: how many of each model's 24 variants beat buy-and-hold after adjustment, against M0's count *(this wording of Q3 is my reading of the plan)* |
| Q4: does accuracy explain value? | Scorecard A (QLIKE, 90% MCS; Entry 4) on the primary cell of the screened book. Then the Spearman correlation of the QLIKE ranking with (a) the primary cell's Sharpe ranking and (b) each model's median Sharpe gain over M0 across the 24 cells. With eight models it is reported, not tested |

### 8. The other books (none changes a choice)

| Book | Rule | Role |
|---|---|---|
| ETF track: SPY alone | SPY timed with M2, the SPY reference's frozen model, at SPY's costs (Entry 2) | Tradeable secondary. Development window February 1997 to December 2018 |
| ETF track: SPY + MTUM tilt | Phase 4's ETF recipe (inverse-volatility mix), M3, the ETF costs | Tradeable secondary. Development window May 2017 to December 2018 |
| Market factor at SPY's costs, M2 | Phase 6's SPY reference, 1946–2018 | Information: SPY's long-history stand-in |
| Phase 6's fixed book, M3 | Market + momentum + BAB from 1950, the full-sample asset list | Information: what selecting assets in real time changes |

**The ETF track.** No choice is made on ETF data: the frozen paper-factor choices are re-run (inverse-variance timing, cap 1.5, monthly, delay 0).
- Forecasts start at the end of the mix's 12th month, the ETF minimum of Phases 3 and 4 *(my choice: Entry 4's 120 months would leave the MTUM book without a forecast until 2023)*. c_t needs 36 pairs (Entry 3).
- The track is not screened. It holds Phase 4's ETF recipe, which matches what the real-time paper screen admits from 2010 among tradeable stand-ins: market and momentum, since the ETFs for BAB and profitability fail filter 4. MTUM fails filter 5's liquidity rule until the end of 2018 (Phase 3); the track holds it anyway, as Phase 4 did.
- **Entry 1, Section 7, item 5.** The ETF track is reported on the holdout with the primary test's statistic and readings, as the tradeable secondary result. It does not replace the primary's verdict *(my reading of the plan's "the holdout is its real test")*.

### 9. Two engine changes, made before any result

- **Weekly holding periods.** The engine now also takes books indexed by weeks ending on Friday (`Book.freq`). Monthly results are unchanged to the last digit: Phase 6's frozen runs reproduce Sharpe ratios of 1.5665238314864909 (primary, M3) and 0.527152631253471 (SPY reference, M2).
- **A fix found by the mechanics run.** `execution.simulate` checked for missing returns with `g_long[held]`. On a DataFrame that sets every instrument held at zero to missing, so any book with a zero position raised an error. Every Phase 5 and 6 book held all its instruments in every month, so the check never fired. No earlier result changes, because the check could only raise. It now tests `isna() & held`, and a test covers it.

### 10. Information only

- The primary's placebo (1,000 shuffled forecasts).
- Every row of Section 8.
- Q4.

### 11. Expectations, written before the run

- **Power.** At about 276 months the primary test's minimum detectable effect is about 0.4 Sharpe a year. A gain like Phase 6's (0.30–0.36) would most likely read **inconclusive**.
- **The combination test** is where Cederburg et al. found most volatility-managed strategies fail. Its 156 out-of-sample months make it weaker still.

---

## Entry 7 — 2026-10-19: Phase 7 outcome, and what 1984 does

**Status.** Written after the Phase 7 build. Entry 6 is not edited, and no rule changed after the first run.

Two things were corrected before the build. Neither touched the code that produces a result:
- **A test expectation.** It had misdated the first scored month of a simulated book; the code was right.
- **Text in the report.**

Section 3 was added after the results had been seen. It is labelled as such and changes no reading.

### 1. Checks of the run

- **Scored months.** 276, as Entry 6 said: 1984, and 1997 to 2018. The family's common months are the same 276.
- **The engine is unchanged.** Phase 6's fixed book (M3) reproduces 1.5665 against 1.2112, and the market factor at SPY's costs (M2) reproduces 0.5272 against 0.4973.
- **Forecasts.**
  - Monthly forecasts stay within 0.04–30 times M0's.
  - Weekly ones reach about 380 times weekly M0's. The cause is M0 near zero in very calm weeks: the unfitted EWMA reaches the same ratio. All are inside Phase 6's 1,000× bound.
  - GARCH refits: ω never exceeds 7.4% of the mean squared return, and b never drops below 0.78. One mix's refits stop short of the optimizer's tolerance in 70% of months, never at a worse likelihood than their start (Entry 5).
- **A live rebuild** of the primary through the code matches the table to 10 digits.

### 2. Outcome under Entry 6's rules (`reports/phase7_*.csv`, `reports/phase7_evaluation.ipynb`)

**The primary test: inconclusive.**

| Item | Value |
|---|---|
| Sharpe ratio, timed / buy-and-hold | 0.746 / 0.733 |
| Gain, standard error, one-sided p | +0.013, 0.180, 0.471 |
| One-sided 95% bounds; minimum detectable effect | −0.283 to +0.309; 0.448 |
| Volatility, timed / buy-and-hold; mean size; share capped | 5.8% / 7.6%; 0.76; 10% |
| CER (γ = 5), timed / buy-and-hold; bootstrap p | 3.47% / 4.12% (gain −0.65 points); 0.72 |
| Combination test (156 months, 2006–2018) | +0.85 points of CER; p = 0.092 |
| Sortino; maximum drawdown | 1.26 / 1.14; −18.9% / −33.5% |
| Break-even κ | 0.03% |
| Placebo (information) | the real α (+0.81%, t 1.1) beats 86.3% of shuffled forecasts: fails the 95% bar |

**The family of 192.**
- 185 of 192 Sharpe gains are positive, with a median of +0.09.
- **One** raw p is below 5%: M3, inverse-volatility timing, shrunk mean-variance mix, weekly (+0.18, p = 0.045).
- **None survives Romano-Wolf** (smallest adjusted p 0.32) **or Holm** (all 1.0).
- The best variant (M0, downside volatility, shrunk mean-variance, weekly, +0.23) has a deflated Sharpe ratio of 0.67, against the 0.95 bar.
- No CER gain is significant (55 of 192 positive).
- 36 combination tests have a raw p below 5%, but these are not adjusted for multiplicity: Holm's smallest adjusted p is 0.64.
- Every variant cuts the maximum drawdown, and 190 raise the Sortino ratio.

**Q3.** None of the 168 comparisons against M0 has a raw p below 5%. Every model's median gain over M0 is between −0.007 and −0.030.

**Q4.** M6 is the most accurate model, and M0 the least and the only one outside the MCS (p 0.003). The rank correlation of accuracy with value is −0.05 on the primary cell, and −0.07 on the median gain over M0.

**The other books (information).**

| Book | Months | Gain | p | Reading |
|---|---|---|---|---|
| ETF track: SPY alone, M2, after costs | 263 (Feb 1997–Dec 2018) | +0.165 | 0.13 | inconclusive (minimum detectable effect 0.37) |
| ETF track: SPY + MTUM tilt, M3, after costs | 20 (May 2017–Dec 2018) | +0.088 | 0.30 | inconclusive |
| Market factor at SPY's costs, M2 | 870 | +0.030 | 0.32 | inconclusive (as in Phase 6) |
| Phase 6's fixed book, M3 | 817 | +0.355 | 0.0002 | edge (as in Phase 6) |

### 3. Information added after the run: what 1984 does

**How it was found.** The primary's Sharpe gain over 1997–2018 looked large in the report's figures, yet the test read +0.01. Splitting by period explained it.

**The mechanism.**
- In 1984 the screen held the market alone. Buy-and-hold ran at 14% volatility, and the timed run at 19% (sizes 0.73–1.5 after a calm 1983).
- From 1997 the book was a diversified mix, with buy-and-hold at 2–9% volatility in most years (14% in 2000–2001).
- A Sharpe ratio pools all months, weighting each by its variance, so the 12 market-only months carry about as much variance as the 264 that follow.
- In them the timed run lost 9.4% a year against buy-and-hold's 4.8%.

| Months | Primary gain (p) | Variants: raw p < 5% / Romano-Wolf / Holm | Phase 6's fixed book, same months |
|---|---|---|---|
| All 276 (the pre-registered test) | +0.013 (0.47) | 1 / 0 / 0 | +0.61 |
| 1997–2018 only (264) | +0.304 (0.012) | 165 / 65 / 2 | — |

The cut was chosen after seeing the result, so it is a diagnosis and not evidence; Section 2's readings stand. It shows two things:
- **The Phase 7 verdict rests on one year** of a book that differs from the rest in kind.
- **The years are not what removed Phase 6's edge.** The fixed book gains +0.61 on the same 276 months, so the real-time asset list did.

### 4. Open before the holdout opens

9. **Years whose book is a single asset.** In the holdout, a year-end screen could again leave a one-asset book, and that year would dominate the test the same way. The options, to be fixed by the user in a dated entry before Phase 9:
    - (a) keep the rule as is;
    - (b) score only years whose screen admits at least two assets (treated like an empty year otherwise);
    - (c) weight each holding year equally, for example by scaling its returns by buy-and-hold's volatility over the past 10 years, known at the year-end.

    Options (b) and (c) are new rules written after this result, and would be recorded as such.
10. **The holdout's book.** Entry 1, Section 6 names the primary's assets as market, momentum and BAB; the user's call in Entry 6 makes the primary's book the real-time screen. The 2018 year-end screen admits exactly those three, so the two agree for 2019, but later year-ends could differ. Confirm that the final test continues the real-time screen.

---

## Entry 8 — 2026-10-19: Phase 8, the robustness checks and the crisis case studies

**Status.** Written before any Phase 8 result. Five things were known when it was written:

- Phase 7's results and Entry 7's diagnosis. 1984, a market-only year, decides the primary's test; over 1997–2018 alone it gains +0.30; Phase 6's fixed book gains +0.355 over its 817 months and +0.61 on the primary's 276.
- Phase 7's figures. The primary's timed path falls less than buy-and-hold in 1998–2000 and 2008, and SPY timed alone falls less in 2000–2002 and 2008.
- Phase 6's results on the fixed book. Its break-even cost is 0.64% one-way, against the primary's 0.03% (Phase 7).
- The plan's Phase 8 table: its checks, their variations and the four crises.
- A mechanics-only run of every Phase 8 book and change on the real panel. It printed dates, counts and the crises' peak and trough dates, and no return statistic (Section 7).

Every rule marked *(my choice)* was set with this entry by Claude, not by the plan or the user.

**Resolves** the Phase 8 items of Entry 1, Section 6 (caps of 1.0, 2.0 and none; timing each piece before combining) and Entry 3, Section 1 (the 1% short-fee stress). **Leaves open** Entry 7's items 9 and 10, which are the user's calls before Phase 9.

### 1. What the checks run on

| Base | What | Role |
|---|---|---|
| **Primary** | Entry 6, Section 3: the screened book, inverse-volatility mix, M3 GJR-GARCH, inverse-variance timing, monthly, cap 1.5, real-time c_t from 120 pairs, no costs. Its 276 scored months | the chosen specification |
| Primary, 1997–2018 | The same runs, scored on 1997–2018 only (264 months): Entry 7, Section 3's cut | information *(my choice)*. 1984 holds the market alone and swamps every pooled statistic (Entry 7), so each check is also read without it. The cut was chosen after Phase 7's result was seen; it is reported beside the primary, never instead of it |
| Fixed book | Phase 6's fixed book: market, momentum and BAB on Phase 4's recipes, scored December 1950 to December 2018 (817 months), at the same specification | information *(my choice)*. The plan's checks were written for this book: its subperiods start in 1926, and its leave-one-out names momentum and BAB |

Every result is an upper bound: paper factors, and no costs outside the cost check.

### 2. The checks: one change at a time

| Check | Base | Variations | Rule |
|---|---|---|---|
| Subperiods | all months | 1926–1962, 1963–1990, 1991–2018 (Phase 2's eras) | The base run, scored on each era's months. Nothing is re-estimated: the walk-forward is already real-time. An era with fewer than 60 scored months is reported without a reading *(my choice)*. The primary scores nothing before 1984 (Entry 6's ten-year training windows), so its first era is empty and its second holds only 1984 |
| Rebalance frequency | monthly | weekly, daily | Weekly: Entry 6, Section 4's weekly M3 (5-day forecast from the month-end fits; c_t from 520 pairs). Daily *(my choice of details)*: trade at every close; M3's one-day-ahead variance σ²_{d+1} from the latest month-end fit, filtered to that close; c_t from 2,520 daily pairs (ten years of 252 days); a recipe takes over on the first trading day after its year-end. Weekly and daily runs are scored on calendar months, and a month counts only if the run holds a position on every trading day of it |
| Estimation window | expanding | rolling 10 years, rolling 20 years | The two estimates the strategy makes at each rebalance: the GARCH fit (on the last 2,520 or 5,040 trading days, the recursion started at the window's mean r², with the warm start and Entry 5's safeguards) and c_t (the last 120 or 240 monthly pairs, at least 120). The screen (expanding) and the mix weights (the last 10 years) keep their own windows: they decide what is held, not how much *(my choice)* |
| Leverage cap | 1.5 | 1.0, 2.0, none | c_t is unchanged: it matches the uncapped volatility |
| Timing rule | inverse variance | inverse volatility, downside inverse volatility | Entry 6, Section 4's definitions, under M3 |
| Combine order | combine, then time | time each, then combine | Each asset in the recipe gets its own size, min(c_i,t / σ̂²_i,t, 1.5), from M3 on its own daily returns (first forecast at its 120th month) and its own real-time c_i,t (120 pairs). Its position is w_i × size_i, put on at the month-end close and left to drift. The benchmark is unchanged. A month is scored once every asset held has a size *(my choice of details)* |
| Leave one out | none dropped | each asset in turn | **Primary:** each factor the screen admits on the development sample (market, momentum, profitability, BAB) is removed from every year-end's list, and the recipe is rebuilt on the rest (Phase 4's Steps 1–2). A year left empty is cash and not scored. **Fixed book:** market, momentum or BAB is dropped, and Phase 4's paper recipe is rebuilt on the other two. Dropping BAB gives Phase 4's paper market + momentum track, the ETF track's paper twin |
| Costs | 0 bp | 5, 10, 20, 30 bp; 30 bp with borrowing | κ one-way per unit traded on every instrument, in both runs. The last row adds borrowed cash at rf + 0.5% a year (Entry 2's financing). Every row also reports its break-even κ (Entry 6, Section 5). κ prices trading each factor as a unit, not the trading inside it (Entry 3, Section 1) |
| Placebo | — | — | Phase 7's: 1,000 shuffled forecasts (seeds 0–999). Each recipe's forecast is reordered in time, then sizes and c_t are rerun. It passes if the real spanning α beats 95% of the shuffled ones (Entry 2). Run on each base book and read on each base's months. As information, it is also read on each era with at least 60 months, with the share of shuffled Sharpe gains the real gain beats |

**Months.** Every change is scored on the base's months that the changed run also scores. So no change adds months, and a change that loses months reports how many.

**Each row reports** Entry 3's Sharpe gain with its standard error, one-sided p and reading; both volatilities; the mean size and share of months capped; turnover and the break-even κ; the CER gain; both maximum drawdowns; and the spanning α.

**Beside the checks** (information):
- **Each asset's share of buy-and-hold's variance**, w_i (Σw)_i / w′Σw, for the inverse-volatility and equal mixes, from each recipe's own 10-year window. The plan expects momentum to carry most of the risk under equal weights.
- **Entry 3's short-fee stress:** the ETF track's SPY + MTUM book with a 1% SPY short fee, against 0.25%. It has 20 development months.

### 3. How a check is read *(my choice)*

- **No p-value is adjusted.** No row can become the result (Entry 1, Section 6), so each p-value is reported raw, as a measure of distance from zero.
- **Each check gets one of three verdicts** against its base:
  - **holds:** every variation has the base's reading;
  - **sign holds:** some reading differs, but every gain has the base's sign;
  - **flips:** some variation's gain has the opposite sign.
- **Eras without a reading** are left out of the subperiod check's verdict.
- **Sturdy or fragile** is judged check by check, not as one score.

### 4. The crisis case studies

| Episode | Peak: the market's highest close in | Trough: its lowest close after the peak, by the end of |
|---|---|---|
| 1929–1932 | 1929 | 1932 |
| October 1987 | 1987 | 1987 |
| 2000–2002 | 2000 | 2002 |
| 2008–2009 | 2007 | 2009 |

- **Peak and trough** are read on the market's total-return level (excess return plus rf). The 2008–2009 peak is searched in 2007, because the market peaked in October 2007 *(my choice)*.
- **The window** runs from 6 months before the peak to 12 months after the trough (the plan's).
- **Runs shown.** Each of these is shown in an episode if it holds a position on every trading day of the window:
  - the primary;
  - Phase 6's fixed book;
  - momentum alone, under Barroso and Santa-Clara's rule (126-day variance, 12% target, no cap, monthly, as published) and under the frozen rule (M3, real-time c_t, cap 1.5);
  - the market alone, under Barroso and Santa-Clara's rule.
- **Only Barroso and Santa-Clara's rule exists in 1929–1932.** It needs six months of data. Every frozen-rule book needs 20 years: 10 to make its first forecast and 10 more for c_t.
- **Panels.** One figure per episode, day by day, with the plan's four rows:
  - price: buy-and-hold's value, log scale;
  - σ̂: the forecast in force, as annualized volatility, beside the realized 22-day volatility;
  - the size w_t;
  - cumulative return relative to cash, timed against buy-and-hold.
- **The three questions, in numbers:**
  1. **Before or after the worst days?** Take buy-and-hold's five worst days between peak and trough. Compare the size held on them with the mean size over the six months before the peak, and the timed run's loss on them with buy-and-hold's.
  2. **Did it miss the rebound?** Growth over the 12 months after the trough, timed against buy-and-hold, beside the same over the fall (peak to trough), with the mean size in each.
  3. **How much did the cap and costs matter?** The window's growth, over whole calendar months: as run, with no cap, at a cap of 1.0 (and 1.5 for Barroso and Santa-Clara's uncapped rule), and at 30 bp a trade.
- **Momentum's crashes.** July–August 1932 and March–May 2009, when Barroso and Santa-Clara's WML lost 91.59% and 73.42%. Reported: momentum at size 1, both managed versions, and the size at the start of each crash. This project's momentum is French's UMD, not their decile spread, so its losses are different numbers.

### 5. Information only

- **The two information bases:** the primary over 1997–2018, and the fixed book.
- **The placebo by era.**
- **The risk shares and the short-fee stress.**
- **The crisis case studies.**

### 6. Expectations, written before the run

- **The primary's gain is small against its standard error** (+0.013 against 0.18). Many of its checks may flip on noise alone. The fixed book (+0.36, standard error 0.10) is the sharper test of sturdiness.
- **Costs.** The primary's break-even is 0.03%, so its gain should turn negative at the first cost row. The fixed book's break-even is 0.64% one-way, so it should keep a positive gain at 30 bp.
- **The cap.** In Phase 6 the fixed book's Sharpe ratio followed its size, and the 1.5 cap bound in 25% of months. A cap of 1.0 should lower its gain; whether any edge is left without leverage is the open question.

### 7. The mechanics run (dates and counts only)

- **Months.** The primary scores 276 months (1984, and 1997–2018), its cut 264, and the fixed book 817. Every change scores all of its base's months, with two exceptions:
  - **Dropping the market from the primary** leaves 1984 empty, so that row scores 264 months.
  - **The weekly primary scores 275.** Its first week of 1997 opens on 27 December 1996, before the 1996 year-end recipe takes over, so January 1997 is not held on every trading day. Phase 7 gave each week to the month of its closing trade and so scored that January; this entry's whole-month rule does not.
- **The daily fixed book** has a c_t from August 1949, earlier than the monthly one. It is scored on the base's 817 months.
- **The crises**, from the market's total-return level:

| Episode | Peak | Trough | Window |
|---|---|---|---|
| 1929–1932 | 3 September 1929 | 8 July 1932 | March 1929 – July 1933 |
| October 1987 | 25 August 1987 | 4 December 1987 | February 1987 – December 1988 |
| 2000–2002 | 24 March 2000 | 9 October 2002 | September 1999 – October 2003 |
| 2008–2009 | 9 October 2007 | 9 March 2009 | April 2007 – March 2010 |

- **The 1987 trough is 4 December, not 19 October.** After the crash the market fell slightly lower in early December.
- **The primary covers only two windows,** 2000–2002 and 2008–2009. It holds cash in 1985–1996 and has no book before 1984.

---

## Entry 9 — 2026-10-19: Phase 8 outcome

**Status.** Written after the Phase 8 build. Entry 8 is not edited, and no rule changed after the first run.

**Two changes after the first build.** Neither touches a check, a verdict or a crisis answer:
- **The risk-share table** (information beside leave-one-out) counted recipes that were never held: 1936's momentum-only recipe, which had no c_t (cash), and the 2018 year-end recipe, held only in the holdout. It now averages over the recipes held in each base's scored years.
- **The crisis paths** now keep two months before each window. This lets the panels' realized 22-day volatility be defined from the window's first day. The answers still use the window only.

Section 3 was added after the results had been seen. It is labelled as such and changes no reading.

### 1. Checks of the run

- **The bases reproduce Phase 7.**
  - Primary: +0.0131 (p 0.471).
  - Phase 6's fixed book: +0.3553 (p 0.0002).
  - The cut without 1984: +0.304 (p 0.012).
- **The placebos reproduce Phase 7's,** with the same seeds: 86.3% for the primary.
- **The daily engine.** Its sizes and returns were rebuilt by hand on 51 days of the fixed book's 1987 recipe and match to 10⁻¹². Its forecasts are unchanged when every return after a date is replaced.
- **The notebook rebuilds the daily fixed-book row** through the code, and it matches the table to 10 digits.
- **Monthly and weekly results are unchanged by the engine's daily periods:** Phase 6's 1.5665238314864909 and 0.527152631253471, and Phase 7's primary and weekly variant.

### 2. Outcome under Entry 8's rules (`reports/phase8_*.csv`, `reports/phase8_robustness.ipynb`)

**Verdicts, each check against its base.** Each cell is the verdict, then how many variations kept a positive gain, how many read "edge", and how many were read.

| Check | Primary (276 months, +0.013, inconclusive) | Primary, 1997–2018 (+0.304, edge) | Fixed book (+0.355, edge) |
|---|---|---|---|
| Subperiods | sign holds (1 of 1 read; edge) | — | sign holds (3 / 2 / 3) |
| Rebalance frequency | holds (2 / 0 / 2) | holds (2 / 2 / 2) | holds (2 / 2 / 2) |
| Estimation window | holds (2 / 0 / 2) | holds (2 / 2 / 2) | holds (2 / 2 / 2) |
| Leverage cap | holds (3 / 0 / 3) | holds (3 / 3 / 3) | holds (3 / 3 / 3) |
| Timing rule | holds (2 / 0 / 2) | holds (2 / 2 / 2) | holds (2 / 2 / 2) |
| Combine order | holds (1 / 0 / 1) | holds (1 / 1 / 1) | holds (1 / 1 / 1) |
| Leave one out | **flips** (3 / 1 / 4) | sign holds (4 / 3 / 4) | holds (3 / 3 / 3) |
| Costs | **flips** (0 / 0 / 5) | sign holds (5 / 2 / 5) | holds (5 / 5 / 5) |

**Selected rows** (Sharpe gain; edge if one-sided p < 0.05):

| Change | Primary | Primary, 1997–2018 | Fixed book |
|---|---|---|---|
| Era 1950/1984–1962 / 1963–1990 / 1991–2018 | – / −0.15 (12 months, not read) / +0.30 | – | +0.03 / +0.19 / +0.57 |
| Weekly / daily | +0.10 / +0.18 | +0.36 / +0.43 | +0.48 / +0.66 |
| Rolling 10 / 20 years | +0.09 / +0.14 | +0.19 / +0.32 | +0.34 / +0.36 |
| Cap 1.0 / 2.0 / none | +0.07 / +0.03 / +0.03 | +0.24 / +0.32 / +0.33 | +0.28 / +0.38 / +0.39 |
| Inverse volatility / downside | +0.08 / +0.04 | +0.22 / +0.28 | +0.29 / +0.30 |
| Time each, then combine | +0.06 | +0.34 | +0.32 |
| Drop market / momentum / profitability / BAB | +0.37 / +0.04 / +0.15 / −0.04 | +0.37 / +0.35 / +0.34 / +0.07 | +0.58 / +0.18 / – / +0.19 |
| 5 / 10 / 20 / 30 bp; 30 bp + borrowing | −0.01 / −0.03 / −0.08 / −0.13; −0.14 | +0.27 / +0.24 / +0.18 / +0.11; +0.10 | +0.33 / +0.30 / +0.25 / +0.19; +0.18 |
| Break-even κ: monthly / weekly / daily | 3 / 7 / 7 bp | 47 / 21 / 13 bp | 64 / 24 / 17 bp |
| Placebo (real α beats) | 86.3%: fails | 99.1%: passes | 100%: passes (by era: 92.2% fails, 97.9%, 100%) |

**Information beside the checks.**
- **Risk shares** (fixed book, averaged over held recipes): market 22%, momentum 53%, BAB 24% under inverse-volatility weights; 56%, 40%, 3% under equal weights. The plan expected momentum to carry most of the risk under equal weights; here the market does.
- **The short-fee stress** (ETF track, 20 months): the gain goes from +0.088 to +0.094.

**Crises** (frozen rule unless marked "B&SC", Barroso and Santa-Clara's rule):

| Episode | Before or after the worst days | The rebound | Cap and costs |
|---|---|---|---|
| 1929–1932 (only B&SC exists) | After: the market held 0.79 into October 1929 and cut to 0.34 for November. On the worst days it held 0.53 and took 54% of the loss | The market missed it: +44% against +180%. Momentum's rebound was its crash: −26% against −52% | A cap of 1.0 would have cost managed momentum 7 points; 30 bp under 1 point |
| October 1987 | After: the fixed book held 0.92 in October and took 91% of the loss; 0.02 in November. Momentum held 1.22 (frozen) and 2.23 (B&SC), and took 1.12 and 2.3 times the loss | Not missed: the fixed book held 1.25 (+1.9% against +2.4%). The market (B&SC) +6% against +20% | Caps of 1.5 or 1.0 would have saved B&SC momentum 2.4 or 5.6 points |
| 2000–2002 | Before: 0.10–0.17 on the worst days, 11–16% of the loss | The mixes rose in the fall (fixed book +29% against +52%) and fell in the rebound (−0.5% against −5.9%) | At most 5 points (30 bp: fixed book) |
| 2008–2009 | Before: 0.17–0.31 on the worst days, 18–29% of the loss | The primary missed part of it (+4.6% against +13.3%). Momentum's crash: −8% against −54%. The market (B&SC) +21% against +75% | At most 3 points; B&SC momentum was levered near 3, and a cap of 1.0 would have cost it 18 points |

**Momentum's crashes** (French's UMD, not Barroso and Santa-Clara's WML):

| Crash | UMD at size 1 | B&SC rule | Frozen rule |
|---|---|---|---|
| July–August 1932 | −46% | −22% (size 0.45) | does not exist yet |
| March–May 2009 | −46% | −15% (size 0.33) | −2% (size 0.06) |

### 3. Information added after the run: daily rebalancing and autocorrelation

- **The pattern.** The fixed book's daily gain over monthly is largest in 1963–1990 (+0.65 against +0.19). That is the era when the mix's daily returns were most autocorrelated: AR(1) 0.38, against 0.24 in 1950–1962 and 0.11 in 1991–2018.
- **The likely reason.** Paper factors built from stale closing prices are autocorrelated like this, and a daily GJR rule that cuts size more after a down day profits from it.
- **What follows.** Part of the daily row's gain is probably not tradeable. The row stands as run; read it as a looser upper bound than the monthly ones.
- **Not a test.** This was found after the result was seen, and no rule was run to separate the two explanations.

### 4. Reading (Entry 8, Section 3)

- **Sturdy:**
  - Phase 6's fixed book. Every check holds or keeps its sign, and it has no edge only in 1950–1962.
  - The primary's 1997–2018 result, provided BAB is held and costs stay below about 10–20 bp.
- **Fragile:**
  - The primary as registered: its reading holds at "inconclusive", but its gain changes sign with any cost or without BAB.
  - Phase 7's verdict still rests on 1984.
- **Unchanged:** Entry 7's items 9 and 10 remain the user's calls before Phase 9.

---

## Entry 10 — 2026-10-19: the holdout's statistic for unequal-risk years, and the holdout's book (the user's calls)

**Status.** These are the user's calls of 2026-10-19 on Entry 7's items 9 and 10. Every rule here was **written after the development results** that motivated it, and **before any holdout data is read**. For the holdout, then, it is a pre-registration. For development it is not: it changes no development result.

**What had been seen when it was written:**
- **Phase 7's primary over all 276 months:** +0.013 (standard error 0.180, p 0.471), inconclusive.
- **The same over 1997–2018 alone (264 months):** +0.304 (standard error 0.134, p 0.012).
- **1984 alone:** −0.148.
- **Phase 7's realized volatility by scored year** (its Figure 10): 19% timed and 14% buy-and-hold in 1984, against 2–9% in most later years.
- **Every Phase 8 result** (Entry 9).

**What had not been seen:** option (c)'s result on development data, and any of its yearly volatilities σ̂_y. Both are computed only after this entry (Entry 11).

**Resolves** Entry 7, items 9 and 10. With this entry, nothing is left open before Phase 9.

### 1. Item 9: years of unequal risk. Option (c), the user's call

**The diagnosis (the user's): the problem is unequal risk, not one asset.**
- 1984 dominates because its book, the undiversified market, carries far more risk than the later factor mixes.
- If 1984's months carried the same risk as the others, adding them should shrink the standard error slightly: 0.134 × √(264/276) ≈ 0.131.
- It rose instead, to 0.180: 1.37 times that. So 4% of the months added 37% to the uncertainty.

**The three options (the user's judgement):**
- **(a) Keep the rule.** It keeps the flaw: whichever book is riskiest decides the test. In a holdout of about 93 months, a one-asset year would be 13% of the data.
- **(b) Score only years with at least two assets.** It counts assets, so it fixes only the one-asset case. A two-asset book where one asset dominates has the same problem.
  - It deletes a year the strategy really traded.
  - On development it is exactly the cut that turns "inconclusive" into "edge", so even sound logic would look like a rescue.
- **(c) Weight each holding year equally.** It fixes the cause: every holding year gets the same risk budget, so no single year can decide the result.

**The rule.**

| Item | Rule |
|---|---|
| Scale | In each holding year y, the monthly excess returns of **both** runs, timed and buy-and-hold, are multiplied by s_y = 1 / σ̂_y |
| σ̂_y | The annualized volatility (√252 × the standard deviation of daily returns) of the book held in year y at size 1: its recipe's weights times its assets' daily returns. It is measured over that recipe's own estimation window, the last 10 years of daily returns up to the year-end, the window its weights come from (Phase 4, Entry 6) *(my choice of window)*. It is known at the year-end |
| Holding year | The months a recipe is held. For the monthly primary, that is the calendar year after its year-end; a partial last year is scaled the same way |
| Statistic, test, readings | Entry 3's, unchanged, applied to the scaled returns: the Sharpe gain, Ledoit-Wolf with a Newey-West covariance (6 lags), one-sided at 5%, the three readings (edge; no meaningful edge if the upper bound is below 0.10; inconclusive) and the minimum detectable effect at 80% power |
| Why it is fair | Both runs in a year are scaled by the same number, fixed at the year-end, so there is no look-ahead and no help to timing. Any constant multiple of s_y gives the same test |

**Its role in the final test.**
- **(c) decides the primary's reading.**
- **(a), Entry 6's unscaled statistic, is reported beside it,** with its own gain, standard error, p and reading.
- **What (c) does not change** *(my choice)*:
  - every other metric of Entry 6 (CER and its test, the combination test, Sortino, drawdown, turnover, break-even, spanning α, the placebo) stays as defined there, on unscaled returns;
  - the ETF track keeps Entry 6's statistic, because its book does not change composition from year to year.

**Development is unchanged.** Phase 7's development verdict stays "inconclusive" under the pre-registered (a), and the dropped-1984 cut stays information, never the headline. Option (c) is computed on development only after this entry, as information (Entry 11).

### 2. Item 10: the holdout's book. The real-time screen, the user's call

- **Confirmed.** The final test continues Entry 6's real-time screen at every year-end, the holdout's included.
- **Entry 6 supersedes Entry 1, Section 6's fixed asset list** (market, momentum, BAB): the later entry wins. The rest of Entry 1, Section 6's primary specification is unchanged (inverse-volatility weights set at each year-end, combine first, the 1.5 cap, Phase 6's forecast), as are Entry 6's rules.
- **Why (the user's):**
  - The screen is part of the frozen procedure; a fixed list would test a different strategy from the one developed.
  - It is not peeking: each year-end screen sees only data up to that date, as a live investor would.
- **Empty years** are cash and not scored (Entry 6). No extra rule is needed: a short scored sample gives a wide band, and the three readings return "inconclusive" on their own.
- **Reported in Phase 9:** the survivor list at each year-end from 2018 (the book held in 2019) to 2025. Changes in it are part of the result.

---

## Entry 11 — 2026-10-19: option (c) on development (information)

**Status.** Computed after Entry 10 fixed option (c), as Entry 10 required. It is information only and changes no rule and no development verdict. Phase 7's pre-registered development result stays inconclusive under (a). Source: `reports/phase8_year_weighted.csv`, `reports/phase8_year_scales.csv` (`python -m src.robustness.build`, step 8).

| Primary, 276 months | Sharpe, timed / buy-and-hold | Gain | Standard error | One-sided p | One-sided 95% bounds | Minimum detectable effect | Reading |
|---|---|---|---|---|---|---|---|
| (a) every month as it is (pre-registered) | 0.746 / 0.733 | +0.013 | 0.180 | 0.471 | −0.283 to +0.309 | 0.448 | inconclusive |
| (c) each holding year scaled to an equal risk budget | 0.881 / 0.666 | +0.214 | 0.137 | 0.059 | −0.011 to +0.440 | 0.341 | inconclusive |

- **(c) also reads "inconclusive" on development.** Its p of 0.059 is just above the 5% level.
- **σ̂_y:**
  - 14.0% for 1984 (the market alone);
  - 4.4–8.2% for the mixes of 1997–2018.
- **1984's share of the pooled variance:**
  - buy-and-hold: 15% under (a), 3% under (c);
  - the timed run: 47% under (a), 13% under (c).
- **Under (c), 1984 is no longer the timed run's largest year; 1997 is (19%).** 1984 still carries about three times an equal share (4%). (c) scales by buy-and-hold's ex-ante risk, and in 1984 the timed run itself levered to 1.5: that is the timing's own decision, which (c) is not meant to neutralize.
- **(c) equalizes ex-ante risk, not realized risk.** 2000 and 2001 (realized buy-and-hold volatility 13–14%, against σ̂ of 4.9%) carry 17–19% of buy-and-hold's variance under (c).
- **(c)'s standard error (0.137) is close to the 1997–2018 cut's (0.134),** as Entry 10's diagnosis implies.
