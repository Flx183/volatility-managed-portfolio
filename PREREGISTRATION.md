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
