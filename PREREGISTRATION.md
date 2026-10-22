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

---

## Entry 12 — 2026-10-19: code-review fixes to Phase 8, the re-run from a rebuilt cache, and the freeze before Phase 9

**Status.** Written after Phase 8's results (Entry 9) had been seen, following a code review of the Phase 8 code (15 findings). No rule of Entries 1–11 is changed. The changes fall into three kinds, kept apart below:
- **bug fixes:** the code did not do what a written rule said;
- **clarifications:** a rule the entries did not state, now stated;
- **additions after the results:** information only.

The state before any change is tagged `phase8-prereview` (commit 99867ee). Each finding has a test that reproduces it (`test/test_review_fixes.py`, `test/test_frozen_rules.py`). Every such test fails on the tagged code and passes on the fixed code. The pinned fingerprints of tables that never moved pass on both, as they should.

### 1. Bug fixes: the code did not do what the written rule said

| Finding | What was wrong | Rule it broke | Effect on reported results |
|---|---|---|---|
| 13 | Phase 8 added a daily entry to `runs.MIN_PAIRS`, which `WF_RULES["min_pairs"]` points to, so Phase 7's frozen table was edited. Entry 8's `daily_min_pairs` was never read: the 2,520 was typed again | Entry 6 (a frozen table); Entry 8 (daily c_t from `daily_min_pairs`) | **None.** Phase 7 has no daily run, and the typed number equalled the registered one |
| 4 | Weekly runs gave each week to the month of its closing trade (Phase 7's rule). So a "whole" calendar month was scored on days shifted by up to four trading days | Entry 8: weekly and daily runs "are scored on calendar months" | **The three weekly rows** (Section 5). No reading or verdict changed |
| 8 | A gain of exactly 0, or a base gain of 0, was called a flip | Entry 8: "flips: some variation's gain has the opposite sign". 0 has no sign | **None:** no gain is exactly 0 |
| 2 | Time-each always used power 1 and an expanding c_t, whatever else the run changed | Entry 8: one change at a time | **None.** No registered row combines time-each with another change; the order rows use inverse variance and an expanding window. The code path was unsafe |
| 1 | The downside rule read weekly origins for a daily book, so every daily downside forecast was missing | Entry 6's downside rule | **None:** no registered row runs the downside rule daily |
| 6 | An asset with no decision yet counted as capped | — | **None:** a mask hid it |
| 5 | A daily book's c_t paired days on the mix's calendar, while its decisions stepped along the panel's | — | **None:** on development there are 0 days where a held asset lacks a return the panel has |
| 9, 10 | A crash in place of a clear message | — | **None** |

**The fixes.**
- **13.** Phase 7's table is restored. Phase 8 builds its own beside it (`checks.MIN_PAIRS`, daily from `ROBUST_RULES`) and passes every rule to the engine from the specification (`checks.resolve`). Phase 7's `runs.run` keeps Entry 6's defaults and refuses a daily book without an explicit minimum.
- **4.** Every Phase 8 run is booked day by day, positions drifting between trades, and each day's return goes to its own calendar month (`src/robustness/ledger.py`). For a monthly book this reproduces the engine's monthly returns bit for bit (tested, and true of every monthly row below). Phase 7's tables keep Phase 7's rule, as Entry 6 registered.
- **8.** Flips compare signs strictly.
- **2.** One specification object (`checks.Spec`) holds all of a run's choices and refuses a second change.
- **1.** Every frequency maps to its own origins, an unknown one is refused, and the forecast's periods must be the origins'.
- **6.** "Capped" is false where an asset has no decision.
- **5.** See Section 2.
- **9, 10.** Guard clauses.

### 2. Clarifications: rules the entries did not state (the user's calls, 2026-10-19)

- **The change in gain is measured on the same months (finding 7).** Before, the table's `change_in_gain` compared a variation's gain on its months with the base's gain on all of the base's months, so a change that lost months mixed two effects. It is replaced by two numbers that add up to it:
  - the **spec effect:** the variation's gain minus the base's gain, on the months both score;
  - the **sample effect:** the base's gain on those months minus its gain on all of its months.
- **Time each, then combine: an asset too short for its c_t (finding 3).** It has no size anywhere. Every period that holds it is cash and not scored, and a segment where no period has a size for every held asset is cash, as Entry 6's "no c_t yet" rule treats the combined mix. On development both orders score the same months. The code used to stop with an error instead.
- **The month of a period's trading cost and turnover (with finding 4)** *(my choice)*. The cost of the trade that opens a period is paid on the period's first day, and the period's turnover, size and capping count in that day's month. For monthly and daily books this is the month they always had.
- **One trading calendar per recipe for daily books (finding 5)** *(my choice of detail)*.
  - A recipe's calendar is the set of days on which every asset it holds has a return. Its c_t pairs on that calendar, and its decision for a day is made at the close of that calendar's previous day. Its daily forecast must cover every day of that calendar from its first.
  - The calendar is set per recipe, not per segment, because each recipe's c_t back-casts over its own mix's history, which a segment-wide calendar would cut.
  - A daily book held on a day its assets have no return now stops, naming the day. On development there is no such day.

### 3. Added after the results (information)

- **The spec effect's interval.** A paired stationary bootstrap draws the months for both runs together, so their common noise cancels. Phase 7's 10,000 draws (seed 0), mean blocks of 6 and 12 months, the larger standard error kept, and the interval spec ± 1.96 se *(my choices)*. Per check, `phase8_checks.csv` counts the intervals wholly above and wholly below 0.
- **Entry 8's verdicts are kept as registered, beside the intervals.** The user asked for the sign verdicts to be replaced by the spec effect and its interval, because with a base gain of +0.013 (standard error 0.180) the primary's sign is noise. Since Entry 8's verdict is a registered reading and the results had been seen, it stays in the tables; the interval is reported next to it *(my choice; the user may drop the verdict column)*. Entry 9's verdicts are unchanged.

### 4. Speed and duplication (findings 11, 12, 14, 15)

- **11.** A daily book builds its trading calendar once. A monthly or weekly decision never builds it. Phase 9's placebo runs 1,000 decisions per recipe.
- **12.** The crisis tables build each run's daily path and monthly returns once, not once per episode.
- **14.** Phase 7's placebo (`walkforward.build.placebo`) now also returns its draws, and Phase 8 reads them instead of running its own copy. A test confirms the alphas are bit-identical to Phase 7's computation, and Phase 7's placebo table and draws are unchanged (Section 5).
- **15, the cache.** It was rebuilt from scratch and diffed (Section 5); no stale entry was found.
  - The code side is deferred until after the holdout, because Phase 9 runs no rolling-window GARCH. Two things remain: the cache key is the hash of whole files, and `variations.garch_fits_window` copies `filters.garch_fits`'s loop to keep that key stable.
  - The key does not include library versions. So `requirement.txt` is now pinned, and an upgrade needs a fresh cache.

### 5. The re-run

**The cache, rebuilt from scratch.** Phase 7's build, Phase 8's build and Phase 6's forecast fill were run with the tagged code on an empty cache.
- **1,978 entries were rebuilt, and every one is bit-identical to the same call in the old cache.**
- **30 old entries were reached by no build** and were not rebuilt. They were single-asset walk-forwards of momentum, BAB and profitability from earlier analyses, and simulated test series.
- **With the tagged code and the rebuilt cache, all 16 Phase 7 and Phase 8 tables equal the committed ones exactly.** So no reported number came from a stale cache.

**The fixed code on the rebuilt cache.**
- **Phase 6:** scorecards A and B and the selection reproduce to 4 × 10⁻¹⁶ (CSV rounding). The placebos were not rerun.
- **Phase 7:** every table and data file is byte-identical.
- **Phase 8: changes, each traced to a fix.**

| Row | Gain before → after | One-sided p | Break-even | Reading | Fix |
|---|---|---|---|---|---|
| Primary, weekly | +0.096 → +0.097 | 0.304 → 0.294 | 7.0 → 7.3 bp | inconclusive | 4 |
| Primary 1997–2018, weekly | +0.359 → +0.348 | 0.0052 → 0.0053 | 20.9 → 21.0 bp | edge | 4 |
| Fixed book, weekly | +0.477 → +0.500 | 2.4 × 10⁻⁵ → 9.9 × 10⁻⁶ | 24.3 → 25.4 bp | edge | 4 |

- **The daily rows** move by at most 10⁻¹², because their calendar months are now compounded through the ledger (4).
- **The crisis answers** move by at most 10⁻¹⁴, because the daily path now runs on the ledger (4).
- **The checks table:** `min_gain` of the two frequency checks whose minimum is the weekly row moves (4).
- **Unchanged:**
  - every monthly row, bit for bit;
  - every verdict;
  - the placebo, the crashes, the risk shares, the short-fee stress and option (c) (Entry 11).
- **Columns:** `change_in_gain` is replaced by `spec_effect`, `spec_se`, `spec_lower`, `spec_upper`, `sample_effect` and `months_lost` (Sections 2–3).

**What the split shows** (spec effect, 95% interval):

| Change | Primary, 276 months | Primary, 1997–2018 | Fixed book |
|---|---|---|---|
| Drop the market | +0.07 (−0.17 to +0.31); sample effect **+0.29** (1984 lost) | +0.07 (−0.17 to +0.31) | **+0.23 (+0.04 to +0.41)** |
| Drop BAB | −0.05 (−0.36 to +0.26) | −0.23 (−0.51 to +0.04) | **−0.17 (−0.32 to −0.02)** |
| Daily rebalancing | **+0.17 (+0.05 to +0.28)** | +0.12 (−0.04 to +0.28) | **+0.31 (+0.14 to +0.48)** |
| Cap 1.0 | +0.06 (−0.08 to +0.20) | **−0.065 (−0.12 to −0.005)** | **−0.07 (−0.13 to −0.01)** |
| 30 bp a trade | **−0.14** | **−0.19** | **−0.17** |

- **The primary's "edge" without the market is 1984's absence.** Its reported +0.36 change in gain is +0.29 from losing 1984 and +0.07 from the change itself, and the +0.07 is indistinguishable from 0.
- **The primary's two "flips" differ.** The cost flip is a real, clearly negative effect. The leave-one-out flip (dropping BAB) is within noise.
- **Most other changes move the gain by less than the paired noise can show.** Every cost row is clearly negative in every base.
- **Entry 9's reading stands:** the fixed book is sturdy; the primary as registered is inconclusive and fragile in sign.

### 6. The freeze before Phase 9

- **Read-only rule tables.** Every frozen rule table outside the forecast-cache key files refuses writes (`src/frozen.py`). Tables inside those files are not wrapped, because any edit there voids every cached forecast.
- **Pinned fingerprints.** Every registered table's fingerprint is pinned at the commit of the phase that fixed it (`test/test_frozen_rules.py`). The only table that had moved was finding 13's.
- **`requirement.txt`** pins the versions that produced Phases 6–8 and the rebuilt cache (Python 3.12.1).
- **The forecast caches are cleared.** Phase 9 starts from an empty cache.
- **For Phase 9.** If a held factor lacks returns inside a holding period, for example because of AQR's publication lag at the end of the data, the monthly engine stops and names the month: `holding_periods` marks the period missing, and `simulate` refuses it. It does not score the month. This was already so; it is recorded here because recent data is where such gaps appear.

---

## Entry 13 — 2026-10-20: Phase 9, the final test on the holdout

**Status.** Written before the holdout is opened; it has never been opened. What was known when it was written:
- every development result (Entries 1–12);
- the holdout's dates and completeness from Phase 1's report: 1,926 trading days, 2 January 2019 to 31 August 2026, no missing day in any series, and AQR's BAB file ending on 31 July 2026, a month before every other series;
- a rehearsal of this entry's build on development data only (Section 9).

The rules in `src/holdout/__init__.py` (`HOLDOUT_RULES`, `EPISODES`, `HOLDOUT_OUTCOME`, `VERDICT`, `DEVELOPMENT`) were written before the rehearsal ran. After it, only the layout of the results table changed (Section 9). Every rule marked *(my choice)* was set with this entry by Claude, not by the plan or the user.

**Resolves** the plan's Phase 9: freeze everything, run once on January 2019 to today (the primary test, the frozen best variant and the ETF track), report the result whatever it is, then the 2020 and 2022 crisis case studies.

### 1. The freeze, and the one run

- **Frozen.** Code, models, parameters and cutoffs are fixed with this entry, before the holdout is opened. Every rule table is read-only and pinned by fingerprint (`test/test_frozen_rules.py`).
- **The run.** `python -m src.holdout.build` opens the holdout through `load_holdout(confirm="phase 9 final test")`, once for the panel and once for the ETF table. It runs once, as the final test.
- **If the build stops after the holdout is opened,** the fix and the reason for reopening go in a dated entry. No rule, model or cutoff changes, and nothing seen in the holdout informs a choice.
- **Forecasts are computed fresh.** The cache is keyed on each series, and the holdout's series are longer than any development series, so no holdout forecast is read from a development run.

### 2. Data

| Item | Rule |
|---|---|
| The holdout | The sealed snapshot, 2 January 2019 to 31 August 2026, checked against its hash when opened. The plan's "today" is read as the snapshot's last day *(my reading)*: new data would change the sealed holdout, which Entry 1 allows only through a dated entry before anything reads it |
| The panel | The development rows, then the holdout's, in one table, so every expanding window runs through the boundary |
| Where the paper books end | The last month in which every factor has a return on every trading day: July 2026, because AQR's BAB ends on 31 July. This holds whether or not BAB is held in 2026 *(my choice: the end is fixed by the data's dates, not by the screen's outcome)* |
| Where the ETF track ends | The last month SPY and MTUM are complete: August 2026 |
| Year-ends | The last trading day of December. Phase 3's `year_ends` would also treat the data's last day as a year-end; it is not one |

### 3. The walk-forward continues

- **Every run is the development walk-forward continued** (Entries 4–6, 10), by the same code on the joined panel. Each decision reads only data up to its own close:
  - the screen at each year-end (filters 1, 3 and 6; level φ; 120 months);
  - the recipe (duplicates, then the mix from the last 10 years);
  - the refit and the forecast;
  - c_t, expanding, with its pairs from all history, development included;
  - the size.
- **The account carries its December 2018 position into January 2019.** January's trade is a rebalance, not an entry from cash. This sets the ETF track's January 2019 cost.
- **Scored months.** Every month from January 2019 that the run holds on every trading day. For the paper books that is January 2019 to July 2026 (91 months), unless a year-end screen admits nothing (cash, not scored: Entry 6). For the ETF track it is January 2019 to August 2026 (92).
- **Every month is scored from the daily ledger** (Entry 12, `src/robustness/ledger.py`). For a monthly book this is the engine's return bit for bit. For the weekly best variant, each day's return goes to its own calendar month *(my choice: Entry 12's fix of finding 4. Phase 7's rule, each week in the month of its closing trade, is used only in the reproduction check of Section 5)*.

### 4. What runs, and what decides each reading

| Book | Specification | Its reading | Role |
|---|---|---|---|
| **Primary** | Entry 6, Section 3: the real-time screened book; inverse-volatility mix; M3 GJR-GARCH; inverse-variance timing; monthly; cap 1.5; c_t from 120 pairs; no costs | **(c)** (Entry 10); (a) beside | **the final test**; an upper bound (paper factors, no costs) |
| Best variant | Phase 7's family member with the largest development Sharpe gain (Entry 6, Section 7's "best"; `reports/phase7_summary.csv`): M0, downside-volatility timing (Entry 6, Section 4), shrunk mean-variance mix, weekly; cap 1.5; c_t from 520 weekly pairs; no costs | (c), with (a) beside *(my choice: its book changes composition the way the primary's does)* | secondary: the plan's "frozen best variant" *(my reading: the only variant the record calls best)*; an upper bound |
| ETF track: SPY alone | Entry 6, Section 8: M2 from the 12th month, c_t from 36 pairs, SPY's costs | (a) (Entry 10) | tradeable secondary |
| ETF track: SPY + MTUM tilt | Phase 4's ETF recipe, inverse-volatility mix, M3 from the 12th month, c_t from 36 pairs, ETF costs with the 0.25% SPY short fee | (a) | tradeable secondary |
| Phase 6's fixed book | Market + momentum + BAB on Phase 4's recipes, the primary's specification. Run on its holdout periods only, from the 2018 year-end's recipe on *(my choice, for compute)*. Each recipe's forecast and c_t come from its mix's whole history, so its decisions equal the full book's | (a) | **information.** Phase 6's development edge (+0.355) meets data it never saw |

- **The statistic** for every book is Entry 3's: the Sharpe gain over buy-and-hold, by Ledoit-Wolf with a Newey-West covariance (6 lags), one-sided at 5%. Its three readings (edge; no meaningful edge if the one-sided 95% upper bound is below 0.10; inconclusive), with the minimum detectable effect at 80% power.
- **(c) and the holding year.** Each holding year's monthly excess returns, both runs, are scaled by 1/σ̂_y (Entry 10; `screened.year_scale` with the book's own mix). The partial 2026 year is scaled like any other. For the weekly variant, a month's scale is its calendar year's.
- **Reported with every book, on the unscaled holdout months:**
  - Entry 6's metrics: both Sharpe ratios and volatilities; CER; Sortino; maximum drawdown; mean; mean size and share capped; turnover; break-even κ; spanning α;
  - the CER test (stationary bootstrap, mean blocks 6 and 12, the larger p);
  - the real-time combination test, trained at each holdout month on every scored month before it, development included *(my choice: a live investor uses all of the past; trained on the holdout alone, 92 months leave no out-of-sample month)*. The SPY + MTUM book (20 development months) and the fixed book (run from 2019) have no out-of-sample month;
  - Lo's (2002) standard error of each Sharpe ratio (information; Section 8);
  - each holding year: its assets, both runs' compounded excess returns and volatilities, the mean size, σ̂_y.
- **The placebo** (information): the primary only, Phase 7's 1,000 shuffled forecasts (seeds 0–999), read on the holdout months.
- **The survivors** at each year-end from 2018 to 2025, with each mix's weights, duplicates dropped and σ̂ (Entry 10, Section 2).

### 5. Checks the build runs

Any failure is reported in `reports/phase9_checks.csv` and the outcome entry, not hidden.
- **Appending the holdout changes nothing before it.** On every month both have, each run is compared with its Phase 7 run:
  - the primary: every numeric column of the engine's output;
  - the best variant, under Phase 7's scoring, except Phase 7's last month, whose last week the development data cut short at 31 December 2018;
  - both ETF books.

  The expected gap is 0.
- **The screen and the recipes.** The survivors and the mix weights at every development year-end must equal Phase 7's.
- **The holdout's hash.** `load_holdout` checks it.

### 6. The 2020 and 2022 crises

| Episode | Trough: the market's lowest close (total return) in | Peak: its highest close from … up to the trough |
|---|---|---|
| 2020 | 2020 | 1 January 2020 |
| 2022 | 2022 | 1 January 2021 |

- **Why a new definition** *(my choice)*:
  - Entry 8's rule (the peak is the highest close of the first year) fails in 2020, when the market ended the year at a new high. So the trough is found first, then the peak before it.
  - 2022's search starts in 2021, as 2008's started in 2007, in case the broad market peaked in late 2021.
  - These are calendar choices. No holdout return was read to make them.
- **The window, the questions and the panels are Entry 8's.** The window runs from 6 months before the peak to 12 months after the trough. The three questions are unchanged:
  - before or after the worst days;
  - the rebound;
  - the cap and costs.

  The panels show four rows: price, σ̂, size, and growth against buy-and-hold.
- **Runs shown,** each if it holds a position on every trading day of the window:
  - the primary, the best variant and the fixed book (as run, no cap, cap 1.0, 30 bp a trade on every instrument);
  - SPY alone and SPY + MTUM (as run, no cap, cap 1.0; their own costs);
  - momentum alone under the frozen rule, and momentum and the market alone under Barroso and Santa-Clara's rule (Entry 8's versions).

### 7. Reading the outcome: the plan's table, fixed here

- **The holdout column** comes from each book's deciding reading:
  - "edge" means **wins**;
  - "inconclusive" means **flat, slightly negative or too short to tell**;
  - "no meaningful edge" means **clearly negative**. With the holdout's standard error near 0.2, this needs a gain below about −0.23.
- **The development column** is "wins" only if the book's own registered development test read "edge":

| Book | Development result (registered) | Development |
|---|---|---|
| Primary | Phase 7: +0.013, p 0.471, inconclusive (Entry 7). Option (c), information: +0.214, p 0.059 (Entry 11) | fails |
| Best variant | Phase 7: +0.234, raw p 0.089, inconclusive; Romano-Wolf p 0.43; deflated Sharpe ratio 0.67 | fails |
| ETF track: SPY alone | +0.165, p 0.13, inconclusive | fails |
| ETF track: SPY + MTUM | +0.088, p 0.30, inconclusive (20 months) | fails |
| Fixed book (information) | +0.355, p 0.0002, edge (Entries 5, 7), on an asset list chosen with the whole development sample | wins |

| Development | Holdout | Verdict (the plan's) |
|---|---|---|
| wins | wins | evidence of a real edge; still modest, given the short holdout |
| wins | flat, slightly negative or too short to tell | inconclusive: too short to tell; report both |
| wins | clearly negative | likely overfit, or the edge faded; report it |
| fails | anything | consistent with Cederburg et al.; report it |

**What this fixes before the holdout is seen.**
- **The primary and every secondary book read "consistent with Cederburg et al."**, whatever the holdout shows, because none won on development. Their holdout numbers are still reported in full, with the minimum detectable effect.
- **Only the fixed book's verdict depends on the holdout,** and it is information.

### 8. Expectations, written before the run

- **Power is low.** The minimum detectable effect is about 0.46 at 92 months (Entry 3, Section 5).
  - The plan's worked example: Lo's standard error of a Sharpe ratio of 0.5 over 92 months is 0.36 a year, so its 95% interval is 0.5 ± 0.71.
  - The Sharpe gain's own standard error is smaller than either Sharpe ratio's, because the two runs move together.
  - In Entry 3's simulations, a true gain like world A's read "inconclusive" 83% of the time at this length.
- **A gain like Phase 6's (+0.36) would most likely read "inconclusive".** "No meaningful edge" needs timing to lose by about 0.2 or more.
- **The primary and the fixed book may coincide.** In any year whose screen admits exactly market, momentum and BAB (as the 2018 year-end screen does), they hold the same recipe: their inverse-volatility weights agree to the last digit on the five development years where both held those three.

### 9. The rehearsal

`python -m src.holdout.build --rehearse` runs this build on development data only and never opens the holdout:
- the pseudo-holdout starts on 1 January 2011;
- BAB's last month (December 2018) is removed, as AQR's file ends a month early in the holdout;
- the crises are 2011 (trough-first, from 1 January 2011) and 2015–2016 (from 1 January 2015).

**What it checked.**
- Every run reproduces its Phase 7 run with a largest gap of exactly 0 on every common month:
  - the primary: 275 months, every one of the engine's 38 numeric columns;
  - the best variant: 276 months;
  - SPY alone: 263; SPY + MTUM: 20.
- The screen's survivors equal Phase 7's at every year-end, and the recipe weights agree to 8 × 10⁻¹⁷.
- The paper books end at November 2018 and the ETF track at December 2018, as the end rules say.
- Runs that do not cover a crisis window are left out and named. The fixed book, run from 2011, is left out of the 2011 window, and SPY + MTUM out of both.

**Its return statistics were printed and seen.** They are development months already inside Phases 7 and 8. They changed no rule:
- primary (c) +0.065 (p 0.37), inconclusive;
- best variant (c) −0.58, no meaningful edge;
- SPY alone +0.088, inconclusive;
- fixed book +0.124, inconclusive;
- placebo 93.0%.

**The one change after it:** the results table no longer repeats the unscaled test beside the `stat_` columns. That repeat carried Phase 8's "too short to read" label for samples under 60 months.

### 10. Information only

- the fixed book;
- the placebo;
- Lo's standard errors;
- the combination test;
- the crisis case studies;
- each holding year's numbers.

None of them changes a reading.

---

## Entry 14 — 2026-10-20: Phase 9 outcome, the final test

**Status.** Written after the one run. Entry 13 is not edited, and no rule changed after the run. Sections 1–5 report the run under Entry 13's rules. Section 6 is information added after the results were seen; it is labelled as such and changes no reading.

### 1. The run, and its checks

- **The command.** `python -m src.holdout.build` ran once, on the code frozen with Entry 13, unchanged. It finished without error in 73 seconds.
- **The opening.** The holdout was opened once, on 2026-10-20 at 03:26:25 UTC (the panel and the ETF table, as recorded in `reports/phase9_summary.csv`), and not reopened. The report notebook reads only the build's saved outputs.
- **Appending the holdout changed nothing before it** (`reports/phase9_checks.csv`). Each run equals its Phase 7 run on every development month, with a largest gap of exactly 0:
  - the primary: 276 months, all 38 numeric columns;
  - the best variant: 276 months, Phase 7's scoring;
  - SPY alone: 263 months; SPY + MTUM: 20 months.

  The screen's verdicts at every development year-end match Phase 7's (0 differences in 651), and so do the recipe weights (largest gap 8 × 10⁻¹⁷).
- **Where the data end.** The paper books end in July 2026 (BAB's last month) and score 91 months, January 2019 to July 2026. The ETF track ends in August 2026 and scores 92.
- **After the run** *(the user's call)*: the git tag and the holdout's access log were removed, and the build's check for them was replaced by a refusal to run again once `reports/phase9_results.csv` exists. No computation changed.

### 2. The holdout's book (Entry 10, Section 2)

| Year-end | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|
| Held in | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
| Survivors | mkt mom bab | mkt mom **rmw** bab | mkt mom bab | mkt mom **rmw** bab | mkt mom bab | mkt mom bab | mkt mom bab | mkt mom bab |
| σ̂_y (inverse-volatility mix) | 7.2% | 4.2% | 9.0% | 5.5% | 9.7% | 9.7% | 9.8% | 9.8% |

- **No duplicate was dropped, and no year was empty.**
- **The primary equals Phase 6's fixed book in six of the eight years.** They differ only in 2020 and 2022, when the screen also admitted profitability.

### 3. The one-shot numbers (`reports/phase9_results.csv`, `reports/phase9_verdict.csv`)

| Book | Statistic | Months | Sharpe, timed / buy-and-hold | Gain | Standard error | One-sided p | One-sided 95% bounds | Minimum detectable effect | Reading |
|---|---|---|---|---|---|---|---|---|---|
| **Primary** | **(c), decides** | 91 | 0.744 / 0.656 | **+0.088** | 0.183 | **0.314** | −0.212 to +0.388 | 0.454 | **inconclusive** |
| Primary | (a) | 91 | 0.897 / 0.855 | +0.041 | 0.182 | 0.410 | −0.257 to +0.340 | 0.451 | inconclusive |
| Best variant | (c), decides | 91 | 0.651 / 0.765 | −0.114 | 0.180 | 0.736 | −0.411 to +0.183 | 0.449 | inconclusive |
| Best variant | (a) | 91 | 0.844 / 0.959 | −0.115 | 0.139 | 0.796 | −0.345 to +0.114 | 0.347 | inconclusive |
| ETF track: SPY alone, after costs | (a) | 92 | 0.790 / 0.891 | −0.102 | 0.169 | 0.726 | −0.379 to +0.176 | 0.420 | inconclusive |
| ETF track: SPY + MTUM tilt, after costs | (a) | 92 | 0.482 / 0.468 | +0.015 | 0.179 | 0.467 | −0.279 to +0.309 | 0.444 | inconclusive |
| Fixed book (information) | (a) | 91 | 0.857 / 0.777 | +0.080 | 0.170 | 0.319 | −0.200 to +0.360 | 0.424 | inconclusive |

**Reported with them** (Entry 6's metrics, on the unscaled holdout months):

| | Primary | Best variant | SPY alone | SPY + MTUM | Fixed book |
|---|---|---|---|---|---|
| Volatility, timed / buy-and-hold | 3.0% / 6.4% | 2.9% / 6.9% | 13.0% / 16.5% | 4.9% / 8.7% | 3.3% / 7.0% |
| Mean size; share of months capped | 0.47; 0% | 0.41; 1% | 0.98; 21% | 0.69; 0% | 0.49; 0% |
| CER (γ = 5), timed / buy-and-hold; CER test p | 2.5% / 4.4%; 0.98 | 2.2% / 5.4%; 1.00 | 6.0% / 7.9%; 0.75 | 1.8% / 2.2%; 0.61 | 2.5% / 4.2%; 0.93 |
| Maximum drawdown, timed / buy-and-hold | −2.8% / −6.8% | −3.7% / −10.3% | −15.0% / −24.3% | −11.3% / −17.8% | −3.0% / −9.0% |
| Turnover a year; break-even κ | 2.7; 5 bp | 9.2; 0 | 3.9; 0 | 2.6; 11 bp | 2.7; 11 bp |
| Spanning α, a year (t) | +0.63% (0.9) | +0.27% (0.3) | +0.54% (0.2) | +0.46% (0.4) | +0.81% (1.1) |
| Combination test: CER gain (p) | −0.03 points (0.52) | −2.1 points (0.87) | −1.2 points (0.74) | no out-of-sample month | no out-of-sample month |
| Lo's standard error of each Sharpe ratio, timed / buy-and-hold | 0.37 / 0.37 | 0.37 / 0.37 | 0.37 / 0.37 | 0.36 / 0.36 | 0.37 / 0.37 |

- **The placebo** (information): the primary's real α beats 69.5% of 1,000 shuffled forecasts on the holdout months, so it **fails** the 95% bar. Its real Sharpe gain beats 77.2% of the shuffled gains.

### 4. The 2020 and 2022 crises (`reports/phase9_crises.csv`)

| Episode | Peak → trough (market fall) | Before or after the worst days | The rebound | Over the window, timed / buy-and-hold |
|---|---|---|---|---|
| 2020 | 19 February → 23 March 2020 (−34.2%) | **Before most of them.** March's size was set at the 28 February close, after the first week of selling. On buy-and-hold's five worst days: primary 0.28 (22% of the loss), SPY 0.25 (24%), best variant 0.05 (4%) | **Missed.** SPY +19% (size 0.67) against +77%; primary +2% (size 0.12) against +12% | SPY +15% / +35%; primary +0.0% / +1.1%; momentum (frozen) +6% / −26% |
| 2022 | 8 November 2021 → 14 October 2022 (−25.6%) | **Before.** Volatility built up first. SPY went from 1.29–1.37 to 0.37 on its five worst days (37% of the loss); the primary held 0.22 (22%) | **Partly missed.** SPY +11% (size 0.88) against +17% | SPY +1.3% / −1.5%; primary +1.6% / +4.4%; SPY + MTUM −10% / −15% |

- **The cap and costs** moved no window by more than about 5 points:
  - SPY with no cap: +16% instead of +15% in 2020;
  - 30 bp a trade: the best variant's 2022 window went from +0.4% to −5.0%.
- **Momentum's crash came in 2020's rebound.** Momentum at size 1 lost 32% in the 12 months after the trough; the frozen rule, at a size of 0.17, lost 7%.

### 5. Reading (Entry 13, Section 7)

| Book | Development | Holdout reading (statistic) | Holdout | The plan's verdict |
|---|---|---|---|---|
| **Primary (the final test)** | fails (+0.013, p 0.471) | inconclusive, +0.088, p 0.31 ((c)) | flat, slightly negative or too short to tell | **consistent with Cederburg et al.; report it** |
| Best variant | fails (+0.234, Romano-Wolf p 0.43) | inconclusive, −0.114, p 0.74 ((c)) | flat, slightly negative or too short to tell | consistent with Cederburg et al.; report it |
| ETF track: SPY alone | fails (+0.165, p 0.13) | inconclusive, −0.102, p 0.73 | flat, slightly negative or too short to tell | consistent with Cederburg et al.; report it |
| ETF track: SPY + MTUM | fails (+0.088, p 0.30) | inconclusive, +0.015, p 0.47 | flat, slightly negative or too short to tell | consistent with Cederburg et al.; report it |
| Fixed book (information) | wins (+0.355, p 0.0002) | inconclusive, +0.080, p 0.32 | flat, slightly negative or too short to tell | inconclusive: too short to tell; report both |

**The final test's verdict: consistent with Cederburg et al. (2020).**
- The primary did not beat buy-and-hold on development, and it does not on the holdout.
- No book reads "edge", and none reads "no meaningful edge". The holdout catches no large failure and supports no edge.

### 6. Information added after the run

- **Why the primary held half of buy-and-hold.**
  - In 2019–2026 the M3 forecast of the mix's volatility averaged 7.8% a year, against 5.3% in 1997–2018.
  - c_t is set on that whole history, so the size averaged 0.47 and never reached the cap. In 2020 and 2022, the years that held profitability, it averaged 0.22 and 0.17.
  - The Sharpe gain does not depend on size. CER and the drawdowns do, which is why the timed run earned less (CER 2.5% against 4.4%) and fell less (−2.8% against −6.8%).
- **Each holdout gain sits below its development gain** under the same statistic, except the primary under (a), whose development gain 1984 had held near zero.
  - The family's best variant fell from +0.234 to −0.115, from the best of 192 to below the primary, as Phase 7's deflated Sharpe ratio (0.67) had warned.
  - The fixed book kept about a quarter of its development edge (+0.080 against +0.355). Its one-sided 95% upper bound (+0.36) just reaches the development estimate.
- **Not a test.** These comparisons were made after the results were seen.

### 7. In plain language

- **Over 91 months out of sample, timing the factor mix by its forecast variance did not reliably beat holding it.**
  - The Sharpe gain was +0.09 a year, against a minimum detectable effect of 0.45.
  - The tradeable SPY version and the best development variant both lost about 0.1.
- **What timing did, it did reliably:** it halved the risk and the drawdowns, and in 2020 and 2022 it cut size before most of the worst days.
- **What it cost:** it held too little in a more volatile holdout, and it missed most of 2020's rebound.
- **The holdout cannot rule out a modest true edge,** but the large development gains did not show up in it.

Every paper result is an upper bound: paper BAB and profitability cannot be traded, and the paper books carry no costs.

---

## Entry 15 — 2026-10-21: after the holdout: three samples, reusing Holdout B, and the sealed forward holdout (Holdout C)

**Status.** Written on 2026-10-21, after Phase 9 opened the holdout (Entry 14) and before any Phase 10 result. It governs every strategy tested from Phase 10 on. The rules are the user's plan of 2026-10-21; those marked *(my addition)* or *(my choice)* were set by Claude with this entry. The rule tables are in `src/retest/__init__.py` (`RETEST_RULES`, `GRADES`, `CONTAMINATION`), read-only and pinned in `test/test_frozen_rules.py`.

### 1. Why new rules

- Phase 9's final test read inconclusive: the primary gained +0.088 (p 0.31) over 91 months, against a minimum detectable effect of 0.45. Every other book read inconclusive too (Entry 14).
- The holdout has now been seen. A new idea tested on it is no longer tested blind, and what Phase 9 showed can leak into the next strategy. Hence the grades, the contamination log and a new sealed sample.

### 2. Three samples

| Sample | Dates | What it is for | Status |
|---|---|---|---|
| Development | 1 July 1926 to 31 December 2018 (VIX from 1990) | EDA, fitting, tuning, choosing | Open |
| Holdout B (reused) | 1 January 2019 to 31 August 2026 | Test only: one run per frozen strategy | Seen in Phase 9 |
| Holdout C (forward) | 1 September 2026 onward | Test only: monthly paper trading | Sealed now |

The boundaries are `config.HOLDOUT_END` and `config.FORWARD_START`.

### 3. Reused as is

- The primary test: the Sharpe gain over buy-and-hold, Ledoit-Wolf with a Newey-West covariance (6 lags), one-sided at 5% (Entry 3).
- Its three readings (edge, inconclusive, no meaningful edge) and the minimum detectable effect at 80% power.
- Romano-Wolf and Holm for families (Entry 6), and the deflated Sharpe ratio.
- The shuffled-forecast placebo.
- The three crisis questions (Entry 8), on Holdout B's 2020 and 2022 episodes (Entry 13).

### 4. Holdout B

- **Test only.** Never fit, tune, screen or choose on it.
  - It loads only through the logged loader: `load_holdout(confirm="holdout B retest")` *(my choice of passphrase)*.
  - Phase 9's passphrase still opens it, but Phase 9's runner refuses to run again.
- **Freeze first, then one run.** Each strategy fixes its code, parameters and a dated entry, pinned to a commit, and only then runs on Holdout B, once.
  - The runner is new: `src/retest/harness.py`. It reuses Phase 9's panel (`src/holdout/build.panels`, which now takes the passphrase as an argument) and writes `reports/phase10_<strategy>_*.csv`. Phase 9's runner and files are otherwise untouched.
  - **Before it opens anything, it refuses** *(my implementation)*:
    - a strategy whose results file (`reports/phase10_<strategy>_results.csv`) exists, so a second run is refused;
    - code that is not frozen. The commit named must be HEAD or an ancestor of it, and its `PREREGISTRATION.md` must hold the strategy's entry. `src/` must equal `src/` at that commit: no change since, committed or not, and no new file.
- **Grades.** Every Holdout B result carries one:
  - **clean**: the idea came from development data or the papers;
  - **exploratory**: the idea came from Phase 9's results (Section 5). An exploratory result is not evidence; only Holdout C can confirm it.
- **Power is unchanged.** About 0.45 a year is the smallest Sharpe gain Holdout B can detect. It can catch a large failure, not prove a small edge.
- **Rehearsal.** Phase 9's rehearsal is reused for every Phase 10 strategy: development data only, a pseudo-holdout from 2011. It never opens the holdout.

### 5. The contamination log: what Phase 9 showed, and what it could bias

| Seen in Phase 9 | Ideas it could bias |
|---|---|
| 2020's rebound missed: SPY timed +19% against +77% | Trend gates, re-entry rules |
| Mean size 0.47; the holdout's volatility above 1997–2018's | Explicit volatility targets, a rolling c_t |
| The downside-volatility variant fell from the best on development to −0.114 | Dropping or keeping downside volatility |
| The screen held market, momentum and BAB at every year-end | The asset list |
| Every Phase 9 book's Sortino ratio, maximum drawdown, CER and volatility (`reports/phase9_results.csv`) *(my addition)* | The choice of a risk metric |
| Every Phase 9 book's monthly returns, timed and buy-and-hold (`data/processed/phase9_runs.parquet`) *(my addition)* | Any variant whose returns are a fixed function of a Phase 9 run, such as a blend of the timed run and buy-and-hold: its Holdout B result can be computed without a new run |

An idea in the second column is graded exploratory on Holdout B.

### 6. Holdout C

- **Sealed from 1 September 2026.** No strategy is run on data after 31 August 2026 until it is frozen.
  - In code *(my implementation)*:
    - the data build puts no row after 31 August 2026 into a holdout file and writes no file for those rows (`src/data/build.split`, `forward_rows`);
    - the split check counts the sealed rows;
    - the loader refuses a table that reaches into Holdout C.
  - The current snapshot ends on 31 August 2026, so nothing in it changed.
  - A `--refresh` today would download Holdout C's months into `data/raw/`. That is allowed; the rule is that no strategy runs on them.
- **Paper trading.** Each frozen strategy is paper traded monthly from its freeze date. Each month's position is logged before the month starts.
- **One formal test date** per strategy, fixed in its entry. The plan proposes 36 months after the freeze.
- **Interim looks.** Every 12 months, and they only check for a large failure (the no-meaningful-edge reading). Peeking for a win is not allowed: it inflates false positives.
- **Open until the first paper book is frozen** *(my addition; not decided here)*.
  - Kenneth French's factors and AQR's BAB are published monthly, weeks after the month ends, and they revise their history.
  - So a paper book's position cannot be logged before the month starts from their data, and a later download changes past months. SPY's position can be logged.
  - Before a paper book is paper traded, its entry says what is logged (the rule's commit and the data vintage) and how new months are appended to the sealed history.

---

## Entry 16 — 2026-10-21: Track A, vol timing as a risk overlay (draft, not frozen)

**Status.** A draft, written before any Track A result on any data.
- No Track A variant has been run on development data, Holdout B or Holdout C. The code in `src/overlay/` is verified on simulated data only (`test/test_overlay.py`).
- Section 6 lists the user's calls before any run. Until every one is made, `require_decided()` refuses (`OPEN` in `src/overlay/__init__.py`).
- This entry may be edited until it is frozen, because no result it governs exists yet. The freeze, its date and its commit will be recorded here.

### 1. The claim

Track A tests what the data already point to: timing improves downside-risk-adjusted returns even when the Sharpe ratio does not move. This is also how the industry uses it.
- **H1.** The timed book's primary risk metric beats buy-and-hold's.
- **H0.** It does not.

### 2. The variants (a family of 3, Holm-adjusted)

| Variant | Rule | Grade on Holdout B (the plan's) |
|---|---|---|
| A0, the frozen books | Phase 9's primary, and SPY with M2, unchanged | Not testable: their risk numbers are printed in Phase 9. Holdout C only |
| A1, rolling volatility target | Size = buy-and-hold volatility over the year's 10-year window / forecast volatility, cap 1.5 | Exploratory (the 0.47 size was seen) |
| A2, fixed blend | Size = 0.5 + 0.5 × c_t / forecast variance | Clean (Wang and Yan); see Section 7 |

How the code reads the rules *(my choices, open to change before the freeze)*:
- **A1's units.** Size = σ̂_y × √(22/252) / √forecast. σ̂_y is a volatility a year and the forecast a 22-trading-day variance, so σ̂_y is read over the same 22 days.
- **A1's σ̂_y.** Set at each December year-end and held for the next year.
  - A screened book: Entry 10's σ̂_y, the buy-and-hold mix's volatility over its recipe's 10-year window (`screened.year_scale`).
  - SPY alone: SPY's volatility over the 10 years up to the year-end.
- **A2's cap.** The managed half is A0's size with its own cap of 1.5, so A2 holds at most 1.25. A2 is then exactly half buy-and-hold and half A0.
- **Months.** A1 and A2 hold in the months A0 holds (those with a c_t), so all three are scored on the same months.
- **Forecasts.** Each book's frozen model: M3 for the screened book, M2 for SPY.

### 3. The primary metric (the user decides before any run)

| Option | Why | Weak spot |
|---|---|---|
| **Sortino gain** (recommended) | Uses every month; scale-free like Sharpe | Still noisy over 91 months |
| Expected shortfall 5%, at matched volatility | Targets the tail directly | Only about 5 tail months in Holdout B |
| Maximum drawdown | What investors feel | One number per sample, almost no power |

- **The Sortino gain test** (`metrics.sortino_gain_test`) is the primary test's machinery applied to another ratio *(my choice: the plan's "reused as is")*.
  - The delta method on (μ_timed, μ_bh, E min(r_timed, 0)², E min(r_bh, 0)²), with the same Newey-West covariance.
  - One-sided at 5%, with the three readings and the minimum detectable effect.
  - Its standard error is within 2% of the spread across 2,000 simulated samples.
- **Matched volatility** *(my addition)*. Any risk number that is not scale-free is compared with buy-and-hold scaled to the timed run's realized volatility over the same months.
  - This covers expected shortfall (as the plan says), and also maximum drawdown and the worst month.
  - A run held at half size, with no timing at all, halves its drawdown, shortfall and worst month. Matched, it wins nothing (`test_holding_less_without_timing_wins_nothing_at_matched_volatility`).
  - Phase 9's primary ran at 3.0% volatility against 6.4%. Scaled to 3.0%, buy-and-hold's −6.8% drawdown is roughly −3.2%, against the timed −2.8%. Most of the halved drawdown in Entry 14 is the smaller size.

### 4. Reported beside it

- Maximum drawdown, expected shortfall 5% and the worst month: each raw, and with buy-and-hold at matched volatility.
- Rebound capture: Entry 8's crisis questions, on the episodes the sample holds.
- CER (γ = 5), both volatilities and the mean size.
- Cost drag: the Sharpe and Sortino gains before costs minus after, and the costs a year.

### 5. The pass rule

- Track A holds if its primary metric reads **edge on Holdout C at the formal date**.
- Holdout B can only drop a variant (a large loss) or give early clean evidence for A2.
- If Track A holds, the deliverable is an overlay function for any long book, and a one-page fact sheet: the volatility cut, the drawdown cut (at matched volatility), and what it costs (missed rebounds, lower CER).

### 6. Open: the user's calls before any run

1. **The primary metric** (Section 3).
2. **Which books.**
   - A0 names two: Phase 9's primary and SPY with M2. If A1 and A2 run on both, the Holm family is 6, not 3.
   - SPY is the only book whose position can be logged before each month starts (Entry 15, Section 6), and it is the kind of long book an overlay is for.
   - For the screened book: whether option (c) decides, as in Entry 10 (my reading: yes).
3. **The no-meaningful-edge bound** on the primary metric's scale. The Sharpe test's 0.10 does not carry over. For symmetric returns a Sortino ratio is about √2 times the Sharpe ratio, so the like-for-like bound is about 0.14.
4. **Holdout C's formal date.** The plan proposes 36 months after the freeze.
5. **A2's grade on Holdout B** (Section 7).

### 7. Notes written before any run *(my additions)*

- **A2 on Holdout B is not blind.**
  - When nothing is charged for size, a run's monthly excess return is linear in its size. The paper books carry no costs, so A2's monthly excess return is exactly 0.5 × buy-and-hold's + 0.5 × A0's (`test_a2_is_exactly_half_buy_and_hold_and_half_a0_when_nothing_is_charged`).
  - Both series are saved for every Phase 9 book (`data/processed/phase9_runs.parquet`), so A2's Holdout B result can be computed without a new run. For SPY, costs make it nearly, not exactly, the same. Nobody has computed it.
  - By the contamination log's last row, A2 would then be exploratory, and Holdout B could give no clean evidence for any Track A variant.
- **Holdout C's power at 36 months.**
  - The Sharpe test's standard error over 91 months was 0.183 (Entry 14). Over 36 months it is about 0.183 × √(91/36) = 0.29, so the minimum detectable effect is about 0.72 a year. In Sortino units it is about 1.4 times that.
  - Detecting a Sharpe-scale gain of 0.2 would take about 470 months (39 years).
  - So within 36 months Holdout C can catch a large failure, but it will rarely confirm a modest risk benefit.

These notes are facts about the design, written before any run. They change no rule here; the user's calls in Section 6 do.

---

## Entry 17 — 2026-10-22: Track B, fixes to the timing rule (draft, not frozen)

**Status.** A draft, written before any Track B result on any data.
- No Track B variant's return, size or test has been seen on development data, Holdout B or Holdout C. The code in `src/fixes/` is verified on simulated data (`test/test_fixes.py`).
- A counts-only check ran every variant end to end on development data for two books (Section 8). It printed no return, size, test or crisis number, and wrote nothing.
- Section 6 lists the user's calls before any run. Until both are made, `require_decided()` refuses, and so does the development build (`OPEN` in `src/fixes/__init__.py`).
- This entry may be edited until it is frozen, because no result it governs exists yet. The freeze, its date and its commit will be recorded here.

### 1. The claim

Track B tries four rule changes, each aimed at a cause found in Phase 9. Each is scored on the Sharpe ratio with the primary test as before (Entry 3). Small gains at best are expected, so "inconclusive" on Holdout B.

### 2. The books

- **Primary (the user decides).** The plan recommends SPY alone after costs: it is tradeable and closest to the paper's market result.
- **Secondary.** The paper factor mix (Phase 9's primary: the real-time screen, M3, option (c)).
- **SPY's main development sample.** The long-history twin: the market factor at SPY's costs, with M2 and c_t from 120 pairs, July 1946 to December 2018 (870 months; Phase 6's SPY reference). SPY itself has 263 development months, February 1997 to December 2018.

### 3. The variants (a family of 4, read through Romano-Wolf and Holm)

| Variant | Rule | Fixes | Source | Grade on Holdout B (the plan's) |
|---|---|---|---|---|
| B1, fixed blend | Size = 0.5 + 0.5 × c_t / forecast variance | Unstable mix weights | Wang and Yan | Clean; see Section 7 |
| B2, downside fixed blend | B1 with downside variance | The same, plus return timing | Wang and Yan | Clean (the downside variant's Phase 9 loss biases against it, not for it) |
| B3, conditional targeting | Size 1 unless the forecast volatility is in the top or bottom 20% of its trailing 10 years; then the usual rule | Noise and turnover in normal months | Bongaerts, Kang and van Dijk (2020) | Clean |
| B4, trend gate | Above the 10-month average, the size never drops below 1; below it, the usual rule | Missed rebounds | The common trend rule | Exploratory: built after seeing 2020 |

How the code reads the rules *(my choices, open to change before the freeze)*:
- **The usual rule** is each book's frozen rule: c_t / forecast variance, capped at 1.5, with the book's own forecast, c_t minimum and costs.
- **B1 and B2's cap.** The managed half is the frozen size with its own cap, so they hold at most 1.25. B1 is Track A's A2 on the same book.
- **B2's downside variance**, read literally: B1's power (1) on the downside forecast. That is the book's forecast × the down-day share of the last 22 days' Σ r² (Phase 7's `runs.downside`), with its own c_t. Phase 7's downside-volatility rule used power 0.5; B2 changes only the forecast.
- **B3's trailing 10 years.** At each month-end, the forecast's rank among its last 120 month-end forecasts, itself included.
  - Top 20% means a rank above 0.8, and bottom 20% a rank of 0.2 or less: 24 of 120 at each end. Ranking the variance or the volatility gives the same order.
  - While fewer than 120 forecasts exist, B3 uses all of them, at least the book's c_t minimum. Only SPY's development months before 2004 have fewer than 120.
  - In a normal month the size is 1 and is not counted as capped.
- **B4's 10-month average.** The book's value at size 1 (total return: excess return plus rf) at each month-end, against the mean of its last 10 month-end values, that one included.
  - Above: the larger of 1 and the frozen size, so at most 1.5.
  - At or below: the frozen size.
- **Timing.** Both signals read the table of the recipe held next month, up to the decision close. That table is back-cast over the mix's whole history, as the forecast and c_t are.
- **Months.** All five runs (the frozen run and the four variants) hold in the frozen run's months, and the family is scored on those.

### 4. Every threshold, fixed now

| Threshold | Value | From |
|---|---|---|
| Blend (B1, B2) | 0.5 at size 1 | the plan; Wang and Yan |
| Extreme states (B3) | the top and bottom 20% | the plan; Bongaerts, Kang and van Dijk (2020) |
| B3's window | 120 month-end forecasts | the plan's "trailing 10 years" |
| Trend (B4) | 10 month-ends | the common 10-month rule |
| Cap | 1.5 | Entry 1 |
| Test | one-sided 5%, Newey-West 6 lags, the three readings | Entry 3 |
| Family | Romano-Wolf (stationary bootstrap, blocks 6 and 12, the larger p, 10,000 draws, seed 0) and Holm | Entry 6 |

No threshold is estimated from data, so step 1's "from development data" holds trivially: none reads Holdout B or C.

### 5. The steps and the pass rule

1. **Fix every threshold** (Section 4), before any run.
2. **Design B4 only on development crises:** 1929–1932, 1987, 2000–2002 and 2008–2009.
   - B4's design is the plan's, unchanged. The crises are read with Entry 8's three questions (`reports/phase10_trackb_dev_crises.csv`).
   - The frozen rule has no c_t on the market before 1946, so the twin covers 1987, 2000–2002 and 2008–2009, not 1929–1932. SPY covers 2000–2002 and 2008–2009.
3. **The development walk-forward.** Each book's family of four goes through Romano-Wolf and Holm (`reports/phase10_trackb_dev_family.csv`).
   - The primary book's development family decides: the twin's if SPY is primary, the paper mix's if it is.
   - The other books are reported beside it.
4. **Freeze, then run once on Holdout B** (`src/retest/`). Each survivor's paper trading starts on Holdout C.

**The pass rule.** A variant moves to Holdout C only if its development gain survives Romano-Wolf (adjusted p below 0.05) and Holdout B does not read "no meaningful edge". Holdout C decides.

### 6. Open: the user's calls before any run

1. **The primary book.** The plan recommends SPY, with development read on the twin.
2. **B1's grade on Holdout B** (Section 7).

### 7. Notes written before any run *(my additions)*

- **B1 is Track A's A2, and its Holdout B result is already implied.**
  - On the paper mix, B1's monthly excess return is exactly 0.5 × buy-and-hold's + 0.5 × the frozen run's (`test_b1_is_half_size_one_and_half_the_frozen_size`). Both are saved for every Phase 9 book (`data/processed/phase9_runs.parquet`). On SPY, costs make it nearly the same.
  - By Entry 15's contamination log, B1 is therefore exploratory on Holdout B, and its run there adds nothing new.
  - If Tracks A and B both keep it, it is one rule tested twice, and it should be counted once.
- **B2's grade is an exception to Entry 15.** The log lists "dropping or keeping downside volatility" as an idea Phase 9 could bias, which would make a downside variant exploratory. The plan grades B2 clean on purpose: Phase 9's downside loss argues against B2, not for it. The exception is recorded here so it is on file.
- **Power on development.**
  - The twin's frozen rule gained +0.030 (p 0.32) over its 870 months (Entry 7). Its standard error is about 0.064, and its minimum detectable effect about 0.16.
  - Surviving Romano-Wolf across four tests needs a t of about 2.2, a gain near 0.14: about five times the frozen rule's gain on the same months.
  - B1 and B2 move half of the frozen size, so their gains are likely smaller than the frozen rule's.
  - **The most likely outcome is that no variant survives on development, so none reaches Holdout C.** SPY's own development months gave the frozen rule +0.165 (p 0.13, minimum detectable effect 0.37; Entry 7).
- **Holdout C's power**, as in Entry 16: at 36 months the minimum detectable Sharpe gain is about 0.72.

These notes are facts about the design, written before any run. They change no rule here; the user's calls in Section 6 do.

### 8. The counts-only check

`python -m src.fixes.build --mechanics --books twin spy`, on 2026-10-22, before any result. It printed only these counts:

| Book | Common months | Every run on exactly those months | Empty cells in the family's rows | Crisis windows covered (episodes × 5 runs) |
|---|---|---|---|---|
| Twin | 870, July 1946 to December 2018 | yes | 0 | 15 of 20 (all but 1929–1932) |
| SPY | 263, February 1997 to December 2018 | yes | 0 | 10 of 20 (2000–2002, 2008–2009) |

The month counts match Phase 7's for both books (Entry 7). The paper mix was not run: it needs M3's walk-forward on every screened mix, 30 to 60 minutes in Phase 7. Its path (several recipes, a cash year) is checked on simulated data (`test_a_screened_book_reads_each_years_recipe_and_skips_its_cash_years`).

---

## Entry 18 — 2026-10-22: Track C, Phase C0: the variance risk premium's EDA and its gate

**Status.** Written before the EDA ran.
- No VRP value, regression or R² has been computed on real data.
- What was known about the new data: the PutWrite file's dates. It has daily values from January 2007, 7 stray rows before, and 1,962 rows after 2018 that were counted, not read.
- The code (`src/vrp/`) is verified on simulated data (`test/test_vrp.py`). The rules are in `src/vrp/__init__.py` (`TRACK_C`, `PUTWRITE`).
- Rules marked *(my choice)* were set by Claude where the plan left a detail open.
- C0 runs once, by `python -m src.vrp.build`. Its outcome goes in the next entry.

### 1. The idea

Track C moves the forecasts to a job where accuracy is the payoff: the gap between the variance options price in (VIX) and the variance we forecast.

VRP_t = (VIX_t / 100)² / 12 − σ̂²_{t+1}

- The first term is the market's implied variance for the next month. VIX covers 30 calendar days (about 21 trading days) and the forecast covers 22: a small mismatch, noted.
- The second term is our forecast of next month's variance, from three models: M0 (rolling realized variance), M4 (HAR) and M6 (HAR-X).
- M6 already uses VIX as an input, so its VRP partly nets VIX against itself. M4 is the cleaner comparison.

### 2. Data

- **Sample.** Development only: the 348 month-ends from January 1990 (VIX's first month) to December 2018.
- **Implied variance.** (VIX/100)² / 12 at the close of each month's last trading day.
- **Forecasts.** Phase 6's walk-forward on the market factor's daily excess returns, each made at the same close for the next 22 trading days.
  - M0 is the last 22 days' Σ r²; M4 is HAR; M6 is HAR-X on M4's inputs (the market fails Test 3; Entry 5).
  - M6's VIX input enters its refits after 60 months of VIX (from 1995), and its event counts after 60 months of the calendar (Phase 6's rule). Before then, M6 runs with fewer inputs.
- **Returns** *(my choice)*. The market factor's monthly excess return, compounded: SPY's long-history twin.
  - SPY starts in February 1993, and the plan's sample starts in 1990.
  - SPY's own months are reported beside the market's, as information; they are not in the gate.
- **h months ahead.** The compounded excess return over months t+1 to t+h: Π(1 + r + rf) − Π(1 + rf).
- **The PutWrite index** (question 5).
  - CBOE's file has daily values only from January 2007. The index's history back to 1986 is not in it, and CBOE's older address no longer serves it. So question 5 reads January 2007 to December 2018.
  - The file is split as Phase 1 splits the panel *(my implementation)*:
    - development rows written;
    - Holdout B rows written and sealed by content hash in `data/putwrite_lock.json`, with the file's URL, fetch time and SHA-256;
    - Holdout C rows not written.
  - C0 never reads the Holdout B rows.

### 3. The questions, and how each is measured

1. **What does VRP look like?** In %² a month: mean, median, standard deviation, the share of months positive, and the AR(1) slope. Also its months around three episodes *(windows my choice)*: June to December 1998, July to December 2001, and July 2008 to March 2009.
2. **Does VRP predict excess returns 1, 3 and 6 months ahead?**
   - The regression is r_{t→t+h} = a + b × VRP_t, by OLS.
   - Newey-West errors use 6 + h − 1 lags *(my choice: the repository's monthly 6, plus h − 1 for the overlap)*.
   - The test is one-sided for b > 0 (the expected sign, after Bollerslev, Tauchen and Zhou 2009), with a normal approximation *(my reading of "the expected sign": the repository's tests are one-sided)*.
3. **Does the sign hold in each of three equal eras?** Each regression's months are split into equal thirds by count. The slope's point estimate must have the expected sign in each. That is the sign rule of the Phase 3 stability filter, with no condition on size or significance.
4. **Does accuracy pay here?**
   - Each model's accuracy is its mean QLIKE (Phase 6's loss) against the next 22 days' realized variance, over the same months.
   - It is set beside its VRP's mean t and mean out-of-sample R² across the three horizons.
   - With three models, this is descriptive: no test.
5. **Does VRP predict put-write returns?** The same regressions, on the PutWrite index's monthly excess returns (its month-end level change minus the month's compounded rf; the index's collateral earns T-bill interest). Information only: no Holm, and not in the gate.

**The out-of-sample R²** (the gate's third condition) follows Campbell and Thompson: R²_OS = 1 − Σ(r − r̂)² / Σ(r − r̄)².
- It uses an expanding window. At each month t, the regression and the benchmark mean are fitted on the pairs complete by t's close (s + h ≤ t).
- The first prediction comes once 60 complete pairs exist *(my choice: Phase 6's minimum for a VIX input)*. That is January 1995 at h = 1.
- The benchmark mean uses the same training window, from 1990, not all of history *(my choice)*.

### 4. The gate (the plan's, written before the EDA is run)

C0 goes to a strategy only if, for at least one forecast and horizon (the same cell), all three hold:
- the predictive slope has the expected sign, with a Holm-adjusted p below 5% across the 9 tests (3 forecasts × 3 horizons);
- the sign holds in all three eras;
- the expanding-window out-of-sample R² on development is positive.

Holm runs across the market's 9 tests only. The SPY and PutWrite tests are information.

### 5. What follows

- **If the gate fails,** Track C stops at C0, and the EDA is reported.
- **If it passes,** C1 follows: the user picks one primary use.
  - Equity timing: SPY's size rises with VRP, combined with volatility scaling, capped between 0 and 1.5. Its main risk is overlap with Track B.
  - Put-write sizing: put-write exposure scaled by VRP and our volatility forecast. Its main risk is short-volatility crashes (February 2018, March 2020).
  - Then C2 to C4: the walk-forward on development with the Phase 5 engine and the primary test; the freeze; one run on Holdout B, graded clean because no VRP result has been seen on it; paper trading on Holdout C.

### 6. Expectations, written before the run *(my additions)*

- **The result is not fully independent of the literature.** Bollerslev, Tauchen and Zhou's sample (1990–2007) lies inside this one. So a pass here would partly re-find a published result. Their VRP also used high-frequency realized variance; ours uses daily data.
- **The gate is demanding.** For the strongest cell to clear Holm across nine tests, its one-sided p must be below 0.05 / 9 ≈ 0.0056.
- **The out-of-sample R² of return predictors is usually small,** and often negative (Welch and Goyal 2008). It is the condition most likely to fail.

---

## Entry 19 — 2026-10-22: Track C, Phase C0 outcome: the gate passes

**Status.** Written after the one run, and Entry 18 is not edited.
- `python -m src.vrp.build` ran once on 2026-10-22, in 3 seconds, on development data only, under Entry 18's rules unchanged.
- The PutWrite file was sealed in the same run (`data/putwrite_lock.json`). Its 1,926 Holdout B rows are hashed, not read. Its 36 Holdout C rows were not written.
- Sections 1–5 report the run under Entry 18. Section 6 is information added after the results were seen; it is labelled as such and changes no reading.
- One bug was found and fixed before the run, by a simulated test: the PutWrite coverage check started at the file's first date instead of that month's first day. No real-data result existed then.

### 1. The gate: passes, in all 9 cells

`reports/phase10_trackc_gate.csv` and `_tests.csv`. The market factor's excess return, January 1990 to December 2018:

| Model | h | Months | Slope t | Holm p | Eras (slope: 1st / 2nd / 3rd third) | Out-of-sample R² |
|---|---|---|---|---|---|---|
| M0 | 1 | 347 | 5.88 | < 0.0001 | + / + / + | 4.4% |
| M0 | 3 | 345 | 6.21 | < 0.0001 | + / + / + | 7.9% |
| M0 | 6 | 342 | 4.03 | 0.0001 | + / + / + | 1.0% |
| M4 | 1 | 347 | 3.58 | 0.0002 | + / + / + | 2.4% |
| M4 | 3 | 345 | 4.82 | < 0.0001 | + / + / + | 5.1% |
| M4 | 6 | 342 | 3.93 | 0.0002 | + / + / + | 8.3% |
| M6 | 1 | 347 | 3.74 | 0.0002 | + / + / + | 1.9% |
| M6 | 3 | 345 | 5.02 | < 0.0001 | + / + / + | 6.2% |
| M6 | 6 | 342 | 3.92 | 0.0002 | + / + / + | 5.2% |

- **Every cell meets all three conditions**, so the gate passes and Track C goes to C1. The eras are January 1990 to August 1999, September 1999 to April 2009, and May 2009 to the end.
- **SPY's own months** (February 1993 on; information) give the same picture: t from 3.3 to 6.0, the sign in every era, and out-of-sample R² from 1.7% to 9.3%.
- **The timing was checked after the run** (Section 6) and is as Entry 18 says. The implied variance is the last trading day's VIX (59.89 on 31 October 2008), the M0 forecast is that month's realized variance, and each target is the return of the months after.

### 2. Question 1: what VRP looks like (`_describe.csv`, `_episodes.csv`)

| Model | Mean (%² a month) | Median | Standard deviation | Months positive | AR(1) |
|---|---|---|---|---|---|
| M0 | 9.1 | 9.3 | 23.5 | 86% | 0.34 |
| M4 | 11.3 | 7.1 | 15.3 | 88% | 0.44 |
| M6 | 11.9 | 8.5 | 10.7 | 99% | 0.66 |

- **Implied variance averages 35.7 %² a month** (a VIX of about 20.7). The forecasts average 23.8–26.6.
- **M6's VRP is positive in 99% of months and the most persistent.** It reads VIX itself, so it nets VIX against itself, as the plan warned.
- **1998 and 2001.** All three VRPs jumped in August 1998 (67–101) and in October 2001 (32–59): implied variance rose faster than the forecasts.
- **2008.** The models part ways.
  - M0's VRP fell to −92 in September 2008 and −245 in October: last month's realized variance exceeded even a VIX of 60.
  - M4's stayed positive (+34 in October), because HAR smooths.
  - M6's was −21 in October.
  - The market then fell another 7.7% in November.

### 3. Question 4: accuracy does not pay here (`_accuracy.csv`)

| Model | Mean QLIKE (rank) | Mean slope t across horizons | Mean out-of-sample R² |
|---|---|---|---|
| M6 | 0.227 (1, most accurate) | 4.23 | 4.5% |
| M4 | 0.256 (2) | 4.11 | 5.3% |
| M0 | 0.344 (3) | 5.37 | 4.4% |

- The least accurate forecast, M0, gives the largest t, and the out-of-sample R² is about the same for all three. Q4's lesson does not flip: a more accurate forecast does not give a stronger predictor.
- Descriptive only: three models, no test.

### 4. Question 5: the PutWrite index, January 2007 to December 2018 (information)

- **In sample, every slope is positive** (t from 1.96 to 5.69, over 137–142 months).
- **Out of sample, it does not hold.** The out-of-sample months run from 2012 only (72–82 months):
  - the R² is negative for M4 and M6 at every horizon, down to −114% (M4, 6 months);
  - M0 gives +4.6% at 1 month, 0.0% at 3 and −0.7% at 6.
- **The sign fails in an era** in 4 of the 9 cells (M0 and M4 at 3 and 6 months).
- Put-write predictability rests on 12 years that include 2008, and it does not carry out of sample.

### 5. Reading

- **The gate passes.** By Entry 18, Section 5, Track C goes to C1, and the user picks one primary use: equity timing or put-write sizing.

### 6. Information added after the run (not a test; it changes no reading)

**What follows comes from the run's own outputs, after its results were seen.**
- **Without July 2008 to June 2009,** every cell keeps its sign in all three eras, with t from 2.5 to 4.9 and out-of-sample R² from 1.4% to 4.3%. The crisis adds strength, but it is not the whole result.
- **The two parts of VRP**, entered separately in one regression:
  - implied variance predicts returns positively (t 3.5 to 5.6);
  - our forecast predicts them negatively (t −2.9 to −5.7);
  - the strength is similar, so the gap between the two carries the information.
  - For equity timing, this matters. Volatility scaling cuts the size when the forecast is high, which agrees with the forecast's part. Implied variance pushes the other way: holding the forecast fixed, a higher VIX predicts higher returns, where volatility scaling would hold less.
- **The third era** (May 2009 to December 2018) lies after Bollerslev, Tauchen and Zhou's sample (1990–2007), and the slope is positive there in every cell. This is the only stretch of the result that is not a re-find of the published one, and it is a sign, not a test.

---

## Entry 20 — 2026-10-22: Step 0 decisions (the user's calls, before any new run)

**Status.** The user's calls on the open items of Entries 15–18, made on 2026-10-22, and written before any run under them.
- **Numbering.** The user's plan names this "Entry 19". That number already holds Track C's C0 outcome.
- **What the user saw.** The user made these calls from my descriptions of Entries 15–19, which were not on GitHub.
- **Step 0.** Entries 15–18 and this entry together are the plan's Step 0: the test rules, the contamination log, the decisions and the Track C gate.
- **What governs.** Where this entry differs from the drafts in Entries 16 and 17, this entry governs. Those entries are not edited.
- **Who set what.** Marked *(my reading)* or *(my choice)* where Claude filled in a detail the calls left open.

### 1. Across all tracks

- **Holdout C's formal test:** 36 months after each strategy's own freeze, for every track. The interim looks of Entry 15 are unchanged: every 12 months, and only for a large failure.
- **Power at that date.** The minimum detectable Sharpe gain is about 0.72 at 36 months. A longer wait does not fix it: at 60 months it is still about 0.56 (0.454 × √(91/60)).
  - So for Tracks B and C, Holdout C can only catch failures.
  - Only Track A can get a confirmation.
- **The record.** Entries 15–18 and this entry are Step 0. The log stays append-only: an entry that governs a result is never rewritten.
- **The out-of-order runs are kept, and recorded here.**
  - Track C's C0 ran on 2026-10-22, before these calls were recorded (Entry 19). It ran on development data only, under Entry 18's gate, which was fixed before it ran. No holdout was touched.
  - Track B's counts-only check (Entry 17, Section 8) also ran before these calls. It printed no result.
- **Commits** are made only when the user says so.

### 2. Track A, the risk overlay (governs over Entry 16)

- **Book:** SPY with M2 only. It is tradeable, its position can be logged before each month, and the family stays small.
- **Variants: A0 and A1. A2 is dropped.**
  - A0 is the frozen SPY rule: c_t / σ̂², capped at 1.5.
  - A1 is the inverse-volatility target (Entry 16).
  - The fixed blend moves to Track B as B1, so the family is two variants, Holm-adjusted.
- **Primary metric: volatility stability.**
  - σ_m is the realized volatility of month m's daily returns, for the timed run and for buy-and-hold, from the daily ledger.
  - R = SD(ln σ_m, timed) / SD(ln σ_m, buy-and-hold), over the scored months. The log makes it scale-free.
  - **Edge** if R < 1, one-sided at 5%, by Entry 6's paired stationary bootstrap (mean blocks 6 and 12, the larger p).
  - **No meaningful edge** if the one-sided 95% upper bound on the reduction 1 − R is below 0.10.
  - **What a pass means:** the overlay keeps risk steadier out of sample than buy-and-hold. It does not mean the overlay earns more, and it will be written that way.
- **Why the rule decides it** (the user's derivation).
  - Let size be s_t = c / σ̂_t^k, x = ln σ̂_t, and ln σ_{t+1} = x + e, with e the forecast error, uncorrelated with x.
  - Then ln σ_{t+1}, timed = ln c + (1 − k)x + e.
  - **Inverse volatility (k = 1, A1)** leaves only Var(e): a large cut.
  - **Inverse variance (k = 2, A0)** leaves Var(x) + Var(e), the same as buy-and-hold. Calm months run hot and wild months run cold, and only the 1.5 cap helps.
  - So A1 is the true overlay. If A0 fails this metric, that is a finding, not a flaw in the test.
- **Secondary:** the Sortino gain (Entry 16's test).
  - Its no-meaningful-edge bound is **0.16**, not 0.14. With a positive mean there are fewer losing months, so Sortino runs above √2 times Sharpe.
  - Phase 7 measured the ratio at 1.26 / 0.746 = 1.69 timed and 1.14 / 0.733 = 1.56 buy-and-hold: about 1.6. 0.10 × 1.6 = 0.16.
- **Fallback.** The first development step is a power check at 36 months. If volatility stability's power there is below 80%, the Sortino gain becomes the primary metric, with the 0.16 bound.
  - **How the power is measured** *(my choice)*: on A1, on development data.
    - The power is Φ(g / (se × √(T / 36)) − 1.645). Here g is the development reduction 1 − R, se its paired-bootstrap standard error (blocks 6 and 12, the larger), and T the development months.
    - It is a normal approximation, like every minimum detectable effect in this file.
  - **The development sample** *(my reading, as in Track B)*: the market factor at SPY's costs with M2, 1946–2018. SPY's own development months are reported beside it.
- **Unchanged from Entry 16:** A1's definition and units, matched volatility for every number that is not scale-free, and the risk numbers reported beside the test.

### 3. Track B, the rule fixes (governs over Entry 17)

- **Primary book: SPY.** Development is read on the twin (the market factor at SPY's costs, 1946–2018). SPY's own months and the paper mix are reported beside it.
- **B1 and B2 scale by inverse volatility, not variance.** The plan wrote "variance", in error.
  - Wang and Yan's equation 1 is f^σ_t = c* / σ_{t−1} × f_t, and their equation 5 uses σ_Down,t−1. Their reported counts out of 103 strategies are for inverse volatility.
  - **B1:** size = 0.5 + 0.5 × c_t / σ̂_t. The managed half has its own c_t and is capped at 1.5, so B1 holds at most 1.25.
  - **B2:** B1 on downside volatility σ̂⁻_t, at power 0.5 with its own c_t: Phase 7's downside-volatility rule, blended. This replaces my power-1 reading in Entry 17.
  - **B3 and B4 are unchanged.** "The usual rule" stays SPY's frozen rule, c_t / σ̂² (power 1).
- **Grades on Holdout B.**
  - B1 is **exploratory**. The switch to inverse volatility is made after Phase 9, and inverse volatility is milder than the rule that lost there, so the change leans in B1's favor.
  - B2 is **clean**: the explicit exception to Entry 15's contamination log, for Entry 17's reason.
  - B3 is clean, and B4 exploratory.
- **B1's grade now rests on that reason alone.** Entry 17's reason (B1's holdout returns follow from Phase 9's saved runs) no longer holds: Phase 9 timed SPY by inverse variance, and B1 now uses inverse volatility.
- **Expectation:** weak development results. The twin's frozen rule read +0.030 in Phase 7.

### 4. Track C, C1: equity timing on SPY, with M4's VRP

- **The rule.** s_t = c_t × μ̂_t / σ̂²_t, capped at 1.5. This is the paper's w = μ / σ², with VRP supplying μ.
  - **μ̂_t** is next month's SPY excess return, forecast from VRP_t (M4) by an expanding-window regression. It uses Entry 18's out-of-sample construction: only pairs complete by t, from 60 pairs on.
  - μ̂_t is shrunk 50% toward the expanding-window mean, at a fixed weight. That is Wang and Yan's lesson: fixed weights beat estimated ones.
  - It is floored at 0 (Campbell and Thompson 2008). *(My reading of the order: shrink first, then floor, so the floor applies to the forecast that sets the position.)*
  - **σ̂²_t** *(my reading)*: SPY's frozen forecast, M2. The only difference from the volatility-only SPY rule is then the VRP term.
  - **c_t:** real-time, as in the engine.
- **The conflict resolves on its own terms.** When VIX rises, both μ̂_t and σ̂²_t rise, and the data decide which wins. If the VRP slope is 0, the rule collapses to Moreira and Muir's.
- **Reported beside the primary test:** the gain against the volatility-only SPY rule (SPY's frozen M2, inverse variance). That isolates what VRP adds.
- **The model, M4.** M4 passed the gate in all three horizons (Entry 19).
  - The choice is recorded after the EDA was seen. It follows Entry 18's statement, made before the run, that M4 is the cleaner comparison.
  - M4 was not the strongest predictor (M0 had the largest t), so the choice does not lean toward the best-looking result.
- **Still to fix in C1's design entry, before its walk-forward:** the development sample (SPY from 1993, or the market twin), the c_t minimum, and the forecast horizon of μ̂ (one month).

### 5. Defaults confirmed

All of Entries 16 and 17's *(my choice)* items are confirmed, with three exceptions:
- **B2's power** is 0.5, not 1 (Section 3).
- **The 1.25 cap** applies to B1 and B2, since A2 is gone.
- **Entry 15's two added contamination rows** are accepted in principle. The user could not see them before this push.

### 6. Expectations, written before any run

- Track A is the only track with a real chance of a confirmed result on Holdout C.
- Tracks B and C will most likely read "inconclusive" on both holdouts.

### 7. A note on the fallback *(my addition)*

**The fallback cannot rescue power.**
- If volatility stability has less than 80% power at 36 months, the Sortino gain will have far less.
- Its smallest detectable gain at 36 months is about 1.1, against a development effect of about +0.12 (Phase 7 primary: 1.26 against 1.14).
- The fallback stands as the user decided it. If the user prefers to keep volatility stability as primary and report its power instead, that change is a new dated entry, made before the power check runs.

---

## Entry 21 — 2026-10-22: Track C, C1's design and C2's rules (the user's calls)

**Status.** Written before C2 ran. No C1 or C2 return, size or test has been computed on real data.
- **What was known:** the C0 gate's results (Entry 19) and Entry 20's calls.
- **The rules** are in `src/vrp/__init__.py` (`C1`), read-only and pinned. The code is in `src/vrp/strategy.py` and `src/vrp/walkforward.py`, verified on simulated data (`test/test_vrp_strategy.py`).
- **The user's calls of 2026-10-22:**
  - μ̂ is estimated on the twin's data.
  - c_t needs 36 pairs.
  - The horizon is one month.
  - **The run is scored on SPY itself, not on the twin.**
- Marked *(my choice)* where Claude filled in a detail.

### 1. The rule (Entry 20)

size_t = c_t × μ̂_t / σ̂²_t, capped at 1.5. This is Moreira and Muir's w = μ / σ², with VRP supplying μ.

### 2. Each piece, its data, and when it starts

| Piece | Data | First value |
|---|---|---|
| VRP's variance term | the twin's M4, in every run | January 1990 |
| μ̂: the regression's fit | the twin's monthly excess returns | first μ̂ at January 1995's close |
| σ̂²: the variance | SPY's frozen M2 (from SPY's 12th month) | January 1994 |
| c_t (36 pairs) and the scoring | SPY's returns, after SPY's costs | first sized month **February 1998** |

- **Why the twin's M4 in every run.** SPY's own M4 would need a 10-year training window, so it would not exist until about 2003.
- **Why the twin only for warm-up.** In Tracks A and B the twin adds 1946–1996. Here VRP starts in 1990, so the twin adds no scored months. Its only job is to warm up μ̂, so the run is scored on the instrument Holdouts B and C will test.
- **The first sized month is February 1998, not January** (correcting the month count given in conversation).
  - The 36 pairs of c_t run from January 1995's signal to December 1997's.
  - The last pair ends with January 1998's return, so c_t first exists at January 1998's close.
  - So the run scores February 1998 to December 2018: **251 months.**

### 3. μ̂, exactly

At each month-end t:
1. Fit next month's twin excess return on VRP by OLS, on the pairs complete by t's close (s + 1 ≤ t), once 60 exist.
2. Forecast at VRP_t.
3. Shrink 50% toward the same pairs' mean, at a fixed weight.
4. Floor at 0, after shrinking (Entry 20).

**This is the gate's own evidence.** The window, the pairs and the mean are exactly those of Entry 18's out-of-sample R². A test checks that the forecasts match it to the last digit. So the gate's evidence covers this rule: M4 at one month passed on its own (Holm p 0.0002, the sign in all three eras, out-of-sample R² 2.4%; Entry 19). No row of μ̂ needs a return after its month.

### 4. How it runs *(my choice of mechanism; the rule is unchanged)*

- **The engine sizes c_t × forecast⁻¹.** C1 hands it σ̂²_t / μ̂_t, which is infinite when μ̂_t is 0, so the size is c_t × μ̂_t / σ̂²_t (0 when μ̂ is 0).
- **Everything else is the engine's own:** the real-time c_t (36 pairs, on SPY's returns), the cap, the costs (5 bp a trade, borrowing at rf + 0.5%) and monthly execution.
- A test checks that the engine's size equals min(c_t × μ̂ / σ̂², 1.5) in every month.

### 5. What is reported

- **The reading:** the primary test (Entry 3) on C1 on SPY. That is the Sharpe gain over SPY held at size 1 after the same costs: one-sided at 5%, the three readings, and the minimum detectable effect. Entry 6's metrics go beside it, along with the share of months at size 0.
- **Beside it** (information):
  - **what VRP adds:** C1 against the volatility-only SPY rule (M2, c_t / σ̂², 36 pairs), both after costs, on C1's months. The Sharpe difference by the primary test's method, and the correlation of their sizes. If VRP's slope were 0, C1 would collapse to that rule;
  - **the twin's run:** the same rule scored on the market factor at SPY's costs, with its own M2. It should nearly match SPY;
  - **the crises:** Entry 8's questions on 2000–2002 and 2008–2009, the development crises these months cover;
  - **the path:** each month's μ̂, its forecast and mean, σ̂², c_t and both sizes.
- **Files:** `reports/phase10_trackc_c2_*.csv`, from `python -m src.vrp.walkforward`.

### 6. Expectations, written before the run

- **"Inconclusive" is the likely reading.** The smallest gain detectable over about 251 months is about 0.4, in line with SPY's 0.37 over 1997–2018 (Entry 7).
- **C2 is not independent evidence.** The EDA chose VRP on these same 1990–2018 months, so a positive development reading would overstate the edge.
- **Holdout B is Track C's first clean test.**

---

## Entry 22 — 2026-10-22: Track C, C2 outcome on development

**Status.** Written after the one run. Entry 21 is not edited, and no rule changed after the run.
- `python -m src.vrp.walkforward` ran on 2026-10-22, in 6 seconds, on development data only.
- Section 6 is information added after the results were seen; it changes no reading.

### 1. Checks of the run

- **The months.** It scored 251 months, February 1998 to December 2018, as Entry 21 said. The first μ̂ is January 1995's.
- **The runs beside it.** The twin's run and the volatility-only rule hold every month C1 on SPY holds.

### 2. The reading: inconclusive (`reports/phase10_trackc_c2_results.csv`)

| Run (251 months) | Sharpe, timed / buy-and-hold | Gain | Standard error | p | One-sided 95% bounds | MDE | Reading |
|---|---|---|---|---|---|---|---|
| **C1 on SPY, after costs (decides)** | 0.563 / 0.371 | **+0.192** | 0.162 | **0.118** | −0.074 to +0.459 | 0.403 | **inconclusive** |
| C1 on the twin, at SPY's costs | 0.562 / 0.393 | +0.170 | 0.155 | 0.138 | −0.086 to +0.425 | 0.386 | inconclusive |
| Volatility-only SPY rule, same months | 0.579 / 0.371 | +0.207 | 0.148 | 0.080 | −0.036 to +0.450 | 0.367 | inconclusive |

The twin's run nearly matches SPY's, as Entry 21 expected.

### 3. What VRP adds: nothing measurable on development (`_vs_volonly.csv`)

- **C1 against the volatility-only rule**, both on SPY after costs, on the same 251 months: Sharpe 0.563 against 0.579. The difference is **−0.015** (standard error 0.097, one-sided p 0.56).
- **Their sizes correlate 0.73.**
- **C1's gain over buy-and-hold on these months is volatility timing.** The volatility-only rule earns as much on its own.

### 4. The rest of Entry 6's metrics: C1 on SPY, against buy-and-hold

| | C1 on SPY | Buy-and-hold | Volatility-only rule |
|---|---|---|---|
| Volatility | 9.4% | 14.9% | 10.5% |
| Mean excess return, a year | 5.3% | 5.5% | 6.1% |
| CER (γ = 5) | 3.1% | 0.0% | 3.3% |
| Maximum drawdown | −30.7% | −56.0% | −31.0% |
| Mean size; share capped; share at size 0 | 0.69; 6%; 3% | 1 | 0.92; 29%; 0% |
| Turnover a year; break-even κ | 4.6; 45 bp | — | 2.7; 85 bp |
| Spanning α, a year (t) | +2.5% (1.9) | — | +3.0% (2.0) |

**C1 held nothing in 8 months:** August 2002, October 2008, September and December 2011, July 2012, November 2014, and February and July 2016. In each, the shrunk forecast was below zero at the month before's close.

### 5. The development crises (`_crises.csv`)

| Episode | Run | Size on buy-and-hold's five worst days (share of its loss) | The fall, timed / buy-and-hold | The rebound, timed / buy-and-hold (mean size) |
|---|---|---|---|---|
| 2000–2002 | C1 on SPY | 0.45 (45%) | −29% / −53% | +12% / +35% (0.48) |
| 2000–2002 | Volatility-only | 0.34 (34%) | −30% / −53% | +13% / +35% (0.55) |
| 2008–2009 | C1 on SPY | 0.08 (7%) | −21% / −56% | +20% / +72% (0.64) |
| 2008–2009 | Volatility-only | 0.16 (15%) | −24% / −56% | +18% / +72% (0.61) |

- **2008.** M4's VRP turned negative at September's close (−21 %² a month), so C1 held nothing in October 2008, when the market fell 17%.
- **Both rules missed most of each rebound.**

### 6. Information added after the run (not a test; it changes no reading)

- **Why VRP adds little to the size.**
  - After the 50% shrink, μ̂ varies far less than σ̂²: its middle half runs from 0.38% to 0.78% a month. So C1's size mostly follows the variance forecast.
  - VRP mattered most at crisis turns. It held nothing in October 2008, against the volatility-only rule's 0.09. In February and March 2009 it held 0.34 and 0.30, against 0.14 and 0.17, as the market bottomed.
- **This is not independent evidence**, as Entry 21 said: the EDA chose VRP on these same months.

### 7. What follows

- The plan's next step for Track C is C3: freeze C1 and run it once on Holdout B. That would be its first clean test, since no VRP result has been seen there. Then paper trading on Holdout C.
- **Whether to freeze C1 as designed is the user's call,** given that VRP added nothing beyond volatility timing on development.

---

## Entry 23 — 2026-10-22: Track C stops after C2

**Status.** The user's call, made after C2, on development data only.
- **Holdout B was never opened for Track C.** No VRP result has been seen on it, and the PutWrite file's Holdout B rows stay sealed and unread.
- Section 3 is information measured after C2's results were seen; it changes no reading.

### 1. What Track C found

- **C0: the gate passed.** VRP predicts the market's excess return 1, 3 and 6 months ahead, in all nine cells (Entry 19).
- **C2: VRP added no value as a sizing signal on top of volatility timing.**
  - C1 against the volatility-only SPY rule, on 251 development months after costs: −0.015, p 0.56 (Entry 22).
  - C1's own gain over buy-and-hold was +0.192 (p 0.12): inconclusive.
- **A side result:** C1 cut the maximum drawdown from −56% to −31%, as the volatility-only rule did (−31%). It supports Track A's subject, but it is not new.

### 2. Why stop here (the user's reasons)

- **Holdout B cannot answer Track C's question**, which is whether VRP adds value on top of volatility timing.
  - On development the answer was −0.015, with a standard error of 0.097 over 251 months.
  - Over Holdout B's 92 months the standard error would be about 0.16 (0.097 × √(251/92)), so it would read "inconclusive" whatever happened.
- **Much of Holdout B's result is already known.** C1's size moves with the volatility-only rule's (correlation 0.73), and SPY's volatility timing lost 0.10 on Holdout B in Phase 9 (Entry 14). The run would largely replay a seen result.
- **Keep Holdout B fresh for VRP.** It can be a clean test for a VRP idea only once. A better use of VRP later, such as put-writing with proper data, would still have it.
- **Fewer looks at the holdout.** Each extra run on Holdout B adds a chance of a fluke pass somewhere across the tracks, and nothing corrects for that.

### 3. Why VRP added nothing: what the run shows (information, after the run)

The explanation proposed in conversation was that μ̂ moves by a factor of about 2 while the variance forecast moves by 36, so the size follows volatility. Measured like for like on C2's 251 months (`reports/phase10_trackc_c2_sizes.csv`), the two spread about equally:

| Spread over C2's months | μ̂ (a month) | σ̂² (SPY's M2) |
|---|---|---|
| Middle half: 75th over 25th percentile | 2.05 (0.38% to 0.78%) | 2.98 (volatility 11.8% to 20.3% a year) |
| 95th over 5th percentile | 9.6 | 10.4 |
| Standard deviation of the log (the 243 months with μ̂ above 0) | 0.57 | 0.75 |

- **The factor of 36 came from volatility's near-full range** (about 10% to 60%; 8.4% to 78.7% in these months), set against μ̂'s middle half.
- **μ̂ and σ̂² rise together.** The correlation of their logs is 0.39: a high VIX lifts both, so they partly cancel in μ̂ / σ̂².
- **So VRP did move the size.** C1's size correlates 0.73 with the volatility-only rule's, not 1. Those moves did not improve the Sharpe ratio over 251 months. This run does not establish why.
- **Not re-run.** Nothing was changed after C2, including the 50% shrink. Changing it now would be tuning on development after seeing the result.
- **Not chased.** VRP's helpful months (October 2008, February–March 2009) were noticed after the fact, on the months the EDA used. No rule is built on them.

### 4. What stays

- Track C's code, its pinned rules (`TRACK_C`, `PUTWRITE`, `C1`) and its results stay in the repository as the record.
- Holdout B stays unseen for any VRP idea.
- **Next:** Tracks A and B continue. Holdout B runs only for their frozen books.

---

## Entry 24 — 2026-10-22: Track B, development outcome

**Status.** Written after the one run. Entries 17 and 20 are not edited, and no rule changed after the run.
- `python -m src.fixes.build` ran on 2026-10-22, in 39 seconds, on development data only.
- The only earlier real-data step was the counts-only check (Entry 17, Section 8).

### 1. Checks of the run

- **The months.** The twin scores 870 months (July 1946 to December 2018), SPY 263 (February 1997 to December 2018) and the paper mix 276 (1984, and 1997 to 2018). Every variant is scored on its book's common months.
- **The frozen runs reproduce Phase 7** (Entry 7): the twin +0.030 (p 0.32), SPY +0.165 (p 0.13), the paper mix +0.013 (p 0.47).

### 2. The decision: the twin's family (SPY is the primary book; Entry 20)

`reports/phase10_trackb_dev_family.csv`, statistic (a), 870 months:

| Variant | Gain | Standard error | p | Romano-Wolf p | Holm p | Reading |
|---|---|---|---|---|---|---|
| B1, inverse-volatility blend | +0.026 | 0.023 | 0.124 | 0.160 | 0.371 | **no meaningful edge** (upper bound +0.064) |
| B2, downside-volatility blend | +0.043 | 0.026 | 0.047 | 0.075 | 0.187 | edge on its own; not after adjustment |
| B3, conditional targeting | +0.028 | 0.053 | 0.297 | 0.262 | 0.371 | inconclusive |
| B4, trend gate | +0.064 | 0.059 | 0.138 | 0.162 | 0.371 | inconclusive |

- **No variant survives Romano-Wolf** (smallest adjusted p 0.075) **or Holm** (0.187).
- **B1 is the closest to a clear answer.** Being half buy-and-hold, it moves with it, so its standard error is small. Its gain is at most about 0.06.

### 3. Beside it (information; neither decides)

| Variant | SPY's own months (263), (a): gain (p; Romano-Wolf p) | Paper mix (276), (c): gain (p; Romano-Wolf p) |
|---|---|---|
| Frozen rule | +0.165 (0.130) | +0.214 (0.059; Entry 11) |
| B1 | +0.066 (0.065; 0.107) | +0.089 (0.005; **0.015**) |
| B2 | +0.066 (0.080; 0.120) | +0.107 (0.004; **0.015**) |
| B3 | +0.066 (0.304; 0.280) | +0.182 (0.077; 0.072) |
| B4 | +0.202 (0.078; 0.120) | +0.238 (0.037; 0.052) |

- **On the paper mix, B1 and B2 survive Romano-Wolf.** The paper mix is a secondary book: an upper bound, since it carries no costs and holds BAB. Its family does not decide.
- **On SPY's own months, no variant survives.** B1, B2 and B3 there are three different runs whose Sharpe ratios happen to agree to three decimals (0.4830, 0.4830, 0.4832), with different means, volatilities and sizes.

### 4. B4 in the development crises (its design; the twin)

| Episode | Rule | Size on buy-and-hold's five worst days | The fall, timed / buy-and-hold | The rebound, timed / buy-and-hold (mean size) |
|---|---|---|---|---|
| October 1987 | Frozen | 0.67 | −23% / −34% | +4% / +20% (0.73) |
| October 1987 | B4 | 0.81 | −27% / −34% | +3% / +20% (0.77) |
| 2000–2002 | Frozen | 0.35 | −33% / −54% | +13% / +39% (0.53) |
| 2000–2002 | B4 | 0.49 | −42% / −54% | +20% / +39% (0.66) |
| 2008–2009 | Frozen | 0.13 | −21% / −56% | +15% / +75% (0.49) |
| 2008–2009 | B4 | 0.13 | −23% / −56% | +33% / +75% (0.82) |

- **B4 did part of what it was built for.** In 2008–2009 it kept the frozen rule's protection on the worst days, and it captured twice the frozen rule's rebound. Even so, that was under half of buy-and-hold's.
- **It did not help in 1987,** a crash out of calm.
- **The trend gate's gain does not survive Romano-Wolf.**

### 5. Reading under the pass rule (Entry 17, Section 5)

- **No variant's development gain survives Romano-Wolf, so no Track B variant moves to Holdout C.**
- Entry 20 expected weak development results, and this is what the run shows.
- **Running Track B on Holdout B could not change this,** and it would add a look at the holdout (Entry 23's reasoning). Whether to run it anyway, or to end Track B at development, is the user's call.

---

## Entry 25 — 2026-10-22: Track A, the volatility-stability measure, its simulated check, and the power check

**Status.** Written before the power check ran. No Track A variant has been run on real data.
- The measure is in `src/overlay/metrics.py` and the power check in `src/overlay/build.py` (`--power`).
- Both are verified on simulated data (`test/test_overlay.py`).
- Marked *(my choice)* where Claude fixed a detail Entry 20 left open.

### 1. The measure, exactly

- **σ_m** = √(Σ r²) over month m's trading days, for the timed account and for buy-and-hold. Each day's r is the account's return relative to cash, from the run's daily ledger *(my choice: the repository's realized-variance convention, not a demeaned standard deviation)*.
- **R** = SD(ln σ_m, timed) / SD(ln σ_m, buy-and-hold) over the scored months. The reduction is g = 1 − R.
- **p:** Entry 6's paired stationary bootstrap. Both columns are resampled by the same months; mean blocks 6 and 12, 10,000 draws, seed 0; the larger p is kept.
- **The standard error** is the larger bootstrap standard deviation of g over the two blocks.
- **Bounds and MDE** *(my choice, as the primary test reads its own)*: the one-sided 95% bounds are g ± 1.645 × standard error, and the minimum detectable effect is 2.4865 × standard error.
- **Readings:**
  - edge if p < 0.05;
  - no meaningful edge if the upper bound is below 0.10;
  - inconclusive otherwise.

### 2. The simulated check: the user's derivation, in a GARCH world

- **The world:** 100 years of daily GARCH(1,1) returns (a = 0.10, b = 0.89, 16% volatility).
- **The size** is c / σ^k on the true expected variance of each month: the sum of the GARCH forecasts of its days' variances, known at the close before. There is no cap and no cost, and the run goes through the engine, with drift inside each month.

| Sizing on | k = 1 (inverse volatility) | k = 2 (inverse variance) |
|---|---|---|
| The true expected variance of the month | **R = 0.674**, edge | **R = 1.001** |
| The next day's variance × the month's days | R about 0.68 | **R about 1.19** |

- **The derivation holds:** ln σ_timed = ln c + (1 − k)x + e.
  - With x the true expected volatility, the error e is uncorrelated with x (correlation about −0.02).
  - Inverse volatility leaves only Var(e): R well below 1.
  - Inverse variance leaves Var(x) + Var(e), as buy-and-hold does: R of 1.
- **Its assumption matters.** A forecast that ignores mean reversion within the month has an error that falls when it rises (correlation about −0.18).
  - Inverse variance on such a forecast overshoots in both directions, and R rises above 1.
  - So A0, the frozen inverse-variance rule on real GARCH forecasts, could read R above 1. That would be a finding.
- **The measure is scale-free.** A run held at half size with no timing gives R = 1 exactly.
- **All three cases are tests.**

### 3. The power check (Entry 20's fallback), as it will run

- **Run:** A1 on the twin. That is the market factor at SPY's costs with M2; A0's months, with c_t from 120 pairs (July 1946 to December 2018); the cap of 1.5; and σ̂_y from the twin's last 10 years at each December year-end.
- **Power at 36 months** = Φ(g / (se × √(T / 36)) − 1.645), with g, se and T from Section 1 on those months.
- **Rule:**
  - if the power is at least 80%, volatility stability stays the primary metric;
  - below 80%, the Sortino gain becomes the primary metric, with the 0.16 bound.
- **The run's development reading** of A1's volatility stability is reported with it. It is an input to the power, not a test of Track A: Holdout C decides.
- **A0 is not part of the power check.**
- **Output:** `reports/phase10_tracka_power.csv`, from `python -m src.overlay.build --power`.

---

## Entry 26 — 2026-10-22: Track A, the power check: the fallback applies

**Status.** Written after the power check ran, once, under Entry 25's rules unchanged.
- `python -m src.overlay.build --power` ran on 2026-10-22, in 3 seconds, on development data only. The output is `reports/phase10_tracka_power.csv`.
- Section 3 is information measured on the same run after its result was seen; it changes nothing.

### 1. The result

| A1 on the twin (870 months, July 1946 to December 2018) | Value |
|---|---|
| R: SD of log monthly volatility, timed over buy-and-hold | **0.940** (0.436 / 0.464) |
| Reduction, 1 − R | 0.060 |
| Standard error; p | 0.046; 0.085 |
| One-sided 95% bounds; MDE | −0.015 to +0.135; 0.113 |
| Development reading | inconclusive |
| **Power at 36 months** | **8.4%** (the bar is 80%) |

### 2. What follows under Entry 20

- **The power is below 80%, so the Sortino gain becomes Track A's primary metric,** with its no-meaningful-edge bound of 0.16.
- Volatility stability is reported beside it.
- **Entry 20, Section 7's note now applies: the fallback does not rescue power.**
  - By the estimate used in conversation (a smallest detectable Sortino gain near 1.1 at 36 months, against a development effect near +0.12), the Sortino gain's power at 36 months is about 9%. That is no better.
  - The estimate is rough: it uses Phase 7's paper primary, not A1 on the twin.
- **A longer wait does not fix it.** For volatility stability to reach 80% power at this development effect would take about 3,200 months (about 260 years).
- **So under either metric, Holdout C at 36 months has under 10% power to confirm Track A.** Entry 20 expected Track A to be the only track with a real chance; on these development estimates it is not.

### 3. Information on the same run (measured after the result; not a test)

- **The real result is far weaker than the simulated one.** The simulation's R was 0.674 with the true volatility and no cap (Entry 25); here R is 0.940.
- **The forecast tracks realized volatility reasonably.** The correlation of A1's log forecast volatility with buy-and-hold's log realized volatility is 0.67.
- **A1's mean size is 1.07, and it sits at the 1.5 cap in 14% of months.**
- **Why R is so much higher than in simulation is not established.** Three candidates exist:
  - the cap, which stops the size rising in calm months;
  - the yearly-moving target σ̂_y;
  - forecast errors that move with the forecast.
  None was tested, and none is chased here: changing the rule now would be tuning after seeing the result.

### 4. The user's call

Track A is now registered with the Sortino gain as its primary metric and very low power at Holdout C. Whether to continue to development runs, the freeze and Holdout B, or to end Track A at development as Tracks B and C did, is the user's call.

---

## Entry 27 — 2026-10-22: Tracks A and B end at development, and Phase 10's testing closes

**Status.** The user's calls after Entries 24 and 26, made on development results only.
- No Phase 10 strategy was frozen. Holdout B was not reopened in Phase 10, and no strategy runs on Holdout C.
- Entry 15's rules stay in force for any future strategy.

### 1. Track B ends at development

- **Nothing passed, so nothing can reach Holdout C.** The pass rule (Entry 17, Section 5) was fixed in advance, and no variant's development gain survived Romano-Wolf (Entry 24). A Holdout B run would only add a look at the holdout.
- **B1's reading is informative in its own right.** Over 870 months, the fixed blend reads "no meaningful edge".
- **B4 is not tuned.** It partly recovered the 2009 rebound (+33% against the frozen rule's +15%), but that was seen after the fact. Adjusting it would fit one crisis.
- **B1 and B2 surviving on the paper mix is recorded as information.**
  - In the user's reading, it fits Wang and Yan, whose evidence is on factor portfolios rather than the market.
  - The paper mix cannot be traded, so it is not tested further.

### 2. Track A ends at development

- **A known-inconclusive test tells nothing.** Its power at 36 months is under 10% under either metric (Entry 26). Thirty-six months of paper trading would buy one more "inconclusive" and one more look at a holdout.
- **No redesign is pursued.**
  - The user's rough scaling: even the simulation's perfect-forecast cut (R = 0.674, a reduction of about 0.33) would need about 9 years at the real data's noise: 263 × (0.06 / 0.33)² ≈ 8.7 years.
  - Real forecasts land between R = 0.67 and 0.94, so A1 is not tuned to chase power.
- **The user's reading, not tested:** forecast error and the 1.5 cap most likely left little of the simulated effect.
- **Optional, and not adopted unless the user asks:** a monthly log of A1's SPY size, labelled a demonstration with no test date. It would not be evidence.

### 3. Phase 10's testing is closed

- **The deliverable is the write-up:** the report's Phase 10 postscript, and the README.
- **Holdout B** was opened once, in Phase 9. Phase 10 did not reopen it for any track.
- **Holdout C** has been sealed since 1 September 2026. No strategy was frozen, so none is paper traded.
- **Every Phase 10 result is on development data.**

### 4. The findings for the write-up

The user's list, with the source of each number made exact *(my additions in italics)*:

1. **VRP predicts market returns** (Entry 19's gate: 9 of 9 cells, 1990–2018), but it added nothing as a sizing signal over volatility timing: −0.015, p 0.56 (Entry 22; *251 months of SPY after costs*).
2. **Four rule fixes did not beat buy-and-hold** (Entry 24): the fixed blend, the downside blend, conditional targeting and the trend gate. *That is on the market factor at SPY's costs over 870 months (1946–2018). Their gains were positive (+0.026 to +0.064), but none survived Romano-Wolf.* The fixed blend reads "no meaningful edge". On the paper mix, B1 and B2 survive (information).
3. **Volatility timing's risk benefit comes in episodes,** not month to month.
   - Month to month, A1 steadied volatility only slightly: R = 0.94 on the twin (Entry 26).
   - In crises the cut was large. *That is the volatility-only SPY rule over 1998–2018 (Entry 22): a maximum drawdown of −31% against buy-and-hold's −56%. Scaled to that rule's volatility (10.5% against 14.9%), buy-and-hold's drawdown is about −40%, so part of the cut is the smaller size.*
   - With a few such episodes a century, no practical forward window can test it.
4. **Inverse volatility steadies risk (R = 0.67); inverse variance does not (R = 1.00).** A forecast that ignores mean reversion within the month makes risk less steady (R = 1.19).
   - *These are simulation results: a GARCH world, sized on the true volatility with no cap (Entry 25). None was measured on real data.*
