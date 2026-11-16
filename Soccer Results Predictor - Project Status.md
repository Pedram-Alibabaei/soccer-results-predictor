# Soccer Results Predictor - Project Status

Status date: 2026-07-06  
Project folder: `C:\Users\Asus ProArt\Desktop\Soccer Results Predictor`

## Executive Summary

We have built a working research/backtesting foundation for a soccer betting assistant. The project can ingest historical soccer results, build pre-match features, train probability models, compare model probabilities against bookmaker odds, simulate value bets, and evaluate closing line value (CLV).

The strongest result so far is the constrained club betting walk-forward test:

| Test season | Bets | Profit | ROI | Hit rate | Positive CLV | Avg decimal CLV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 109 | +1.61 | +1.48% | 54.13% | 52.29% | +0.69% |
| 2023 | 114 | +4.93 | +4.32% | 57.02% | 56.14% | +0.92% |
| 2024 | 111 | +4.08 | +3.68% | 58.56% | 54.05% | +0.19% |
| Total | 334 | +10.62 | +3.18% | 56.89% | 54.19% | +0.60% |

This is promising because profit and CLV were positive across all three tested seasons. It is not yet enough to justify real-money betting. The model is ready for paper testing and further research, not live betting.

## Readiness Verdict

Short answer: the model is **not ready for real-money betting yet**.

It is ready for:

- Offline research.
- Historical backtesting.
- Paper-trading simulations.
- Comparing model-selected opening odds against closing odds.
- Testing whether the current signal survives stricter filters.

It is not ready for:

- Real-money betting.
- Fully automated bet recommendations.
- World Cup betting with paid odds.
- Live operational use.

Why not ready yet:

- The strongest positive result covers only 334 selected bets.
- The edge is modest: total ROI is +3.18%.
- CLV is positive but small: average decimal CLV is +0.60%.
- The current club model uses free Football-Data stats and does not yet include true xG, player strength, injuries, suspensions, or confirmed lineups.
- There is no live odds ingestion yet.
- There is no bankroll/drawdown/staking engine beyond flat staking and a placeholder Kelly helper.
- The profitable experiment is club-league based, not yet transferred to national-team or World Cup betting.
- Paid international odds should not be bought until the framework is more proven.

My recommendation: continue development and paper trade only. Do not use real money yet.

## Original Goal

The project started with the idea of building a program that can help decide what to bet on soccer matches, starting with World Cup/national-team games and later potentially Formula 1.

The desired final system is:

1. Receive upcoming matches and bookmaker odds.
2. Estimate fair probabilities across relevant betting markets.
3. Compare fair probabilities with bookmaker prices.
4. Identify value bets.
5. Build a small set of recommended bets.
6. Avoid longshot traps and over-betting.
7. Track performance by profit, ROI, CLV, hit rate, and drawdown.

The project has moved from simple match prediction toward a more realistic betting workflow:

- Opening odds vs final result.
- Opening odds vs closing odds.
- Residual value filtering.
- Max odds constraint.
- Top 3 bets per club weekend.

## Current Project Structure

Important folders:

```text
Soccer Results Predictor/
  data/
    raw/
    processed/
    external/
  docs/
  examples/
  reports/
  scripts/
  src/soccer_predictor/
  tests/
```

Important source modules:

| File | Purpose |
| --- | --- |
| `src/soccer_predictor/markets.py` | Converts score probabilities into betting-market probabilities. |
| `src/soccer_predictor/elo.py` | Builds chronological Elo and attack/defense features. |
| `src/soccer_predictor/data_pipeline.py` | Downloads/processes international results and World Cup holdouts. |
| `src/soccer_predictor/modeling.py` | Trains softmax, Poisson, Dixon-Coles, and draw-aware Poisson models. |
| `src/soccer_predictor/club_data.py` | Downloads/processes Football-Data club odds CSVs. |
| `src/soccer_predictor/club_features.py` | Adds rolling club features: shots, shots on target, rest days, form. |
| `src/soccer_predictor/odds.py` | Converts odds to no-vig probabilities and evaluates value bets. |
| `src/soccer_predictor/calibration.py` | Blends model probabilities with bookmaker probabilities. |
| `src/soccer_predictor/value_model.py` | Builds residual value candidates, CLV metrics, and bet selection constraints. |

Important scripts:

| Script | Purpose |
| --- | --- |
| `scripts/build_dataset.py` | Build international/national-team dataset. |
| `scripts/evaluate_holdout.py` | Evaluate World Cup tournament holdouts. |
| `scripts/build_club_odds_dataset.py` | Build free club odds dataset from Football-Data. |
| `scripts/backtest_club_odds.py` | Backtest raw model probabilities against bookmaker odds. |
| `scripts/backtest_residual_value.py` | Backtest residual value model and CLV-constrained betting. |

## Data Sources Used

### International/National-Team Data

Source: Mart Jürisoo's international results dataset.

Current processed dataset:

- Raw rows: 49,498
- Completed match rows: 49,487
- Date range: 1872-11-30 to 2026-07-01
- Chronological split:
  - Train: 42,226 rows
  - Validation: 3,581 rows
  - Test: 3,680 rows
- World Cup 2018 holdout:
  - Train rows: 41,639
  - Test rows: 64
- World Cup 2022 holdout:
  - Train rows: 45,700
  - Test rows: 64

Important caveat:

Some World Cup knockout source scores include extra time. We added `data/external/world_cup_settlements.csv` to correct 2018 and 2022 knockout matches for:

- 90-minute 3-way market settlement.
- Qualification/who-goes-through settlement.

### Club Odds Data

Source: Football-Data.co.uk CSVs.

Current processed club dataset:

- Leagues:
  - Premier League
  - La Liga
  - Serie A
  - Bundesliga
  - Ligue 1
- Seasons:
  - 2019-2020 through 2024-2025
- Matches: 10,707
- Date range: 2019-08-09 to 2025-05-25
- Rows with 1X2 odds: 10,707
- Rows with Over/Under 2.5 odds: 10,707

Odds columns include both opening and closing odds where available.

## Features Currently Used

### National-Team Features

- Home/away Elo before match.
- Elo difference before match.
- Rolling attack and defense strength.
- Match count/experience.
- Neutral-site flag.
- World Cup stage/context flags.
- World Cup group/knockout indicators.

### Club Features

The club model now uses:

- Pre-match Elo.
- Rolling goals for/against.
- Rolling shots for/against.
- Rolling shots on target for/against.
- Rest days.
- Recent overall form.
- Home-team home form.
- Away-team away form.
- Opening odds.
- Closing odds for CLV evaluation.

Prepared but not truly populated yet:

- xG features.

Reason: Football-Data does not provide true xG in these CSVs. We intentionally did not fake xG.

## Betting Markets Covered

The score-probability engine supports:

- 3-way result: `1`, `X`, `2`
- Double chance: `1X`, `12`, `X2`
- Over/Under goals
- Both teams to score
- Draw no bet
- Correct score
- European handicap
- Match result plus Over/Under combinations

The current serious backtests focus on:

- 1X2
- Over/Under 2.5

The final app can later expand into the other markets shown in the user's bookmaker screenshots, but those markets still need odds data and settlement validation.

## Models Implemented

### 1. Softmax Baseline

Predicts 1/X/2 directly from pre-match features.

### 2. Poisson Goal Model

Predicts expected home goals and away goals, builds a score matrix, then derives market probabilities.

### 3. Dixon-Coles Poisson Model

Adds a low-score correction to the Poisson score model. This has generally been the best simple base model so far.

### 4. Draw-Aware Poisson Model

Adds a separate draw model on top of the goal model.

### 5. Bookmaker Blend

Learns whether to trust:

- the model,
- the bookmaker no-vig probabilities,
- or a blend of both.

Early result: the blend often chose bookmaker-only, which correctly prevented bad bets when the standalone model was weak.

### 6. Residual Value Model

This is the most betting-relevant model so far.

It does not merely ask:

> Who will win?

It asks:

> Given the model probability, bookmaker probability, odds, match features, and market context, is this bet candidate historically worth selecting?

The residual model is trained using walk-forward out-of-sample candidates, which prevents it from learning on predictions made by a base model that had already seen the match.

## World Cup Results So Far

The World Cup work is useful for probability modeling, but not yet for betting because we do not have historical bookmaker odds for those tournaments.

### Corrected 90-Minute 3-Way Results

World Cup 2018, clean 3-way target:

- Model: Poisson
- Test rows: 64
- Accuracy: 56.2%
- Log loss: 0.951
- Brier: 0.563

World Cup 2022, clean 3-way target:

- Model: Poisson
- Test rows: 64
- Accuracy: 54.7%
- Log loss: 1.055
- Brier: 0.608

Interpretation:

The national-team model has reasonable directional predictive power, but this is not yet a betting model because it is not compared to historical World Cup odds.

### Qualification/Who Goes Through

World Cup 2018:

- Accuracy: 56.2%
- Log loss: 1.047
- Brier: 0.629

World Cup 2022:

- Accuracy: 62.5%
- Log loss: 0.580
- Brier: 0.413

Caveat:

The qualification dataset has a source-order bias. A real betting version must compare against bookmaker-displayed team labels and historical qualification odds.

## Club Betting Results So Far

### Early Pure Model Backtests

Pure model value betting was not good enough.

Examples:

- Poisson model using opening odds on 2024-2025:
  - Selected bets: 2,110
  - Profit: -135.58 units
  - ROI: -6.43%

- Dixon-Coles model using opening odds on 2024-2025:
  - Selected bets: 2,019
  - Profit: -126.12 units
  - ROI: -6.25%

Interpretation:

The raw model found too many fake edges. It was especially vulnerable to weak longshot/underdog picks.

### Bookmaker Blend

The bookmaker blend often chose alpha `0.00`, meaning:

> Trust bookmaker no-vig probabilities, do not let the standalone model override the market.

This was useful as a safety mechanism.

### Residual Value Model

The residual value model reduced betting volume dramatically.

Without constraints, it still struggled:

- Dixon-Coles residual, 2024:
  - Selected bets: 183
  - Profit: -20.89
  - ROI: -11.42%

Adding betting discipline made a big difference:

- Max odds: 4.00
- Max bets per weekend: 3
- Opening odds only
- CLV check against closing odds

## Best Current Result: Constrained CLV Walk-Forward

Command pattern:

```powershell
$env:PYTHONPATH = "src;."
python scripts\backtest_residual_value.py --test-season 2024 --base-model dc_poisson --odds-timing opening --max-odds 4.00 --max-bets-per-weekend 3 --min-base-train-seasons 1
```

Summary:

| Test season | Bets | Profit | ROI | Hit rate | Avg odds | Positive CLV | Avg decimal CLV | Avg no-vig probability CLV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 109 | +1.61 | +1.48% | 54.13% | 2.02 | 52.29% | +0.69% | +0.44% |
| 2023 | 114 | +4.93 | +4.32% | 57.02% | 1.84 | 56.14% | +0.92% | +0.55% |
| 2024 | 111 | +4.08 | +3.68% | 58.56% | 1.79 | 54.05% | +0.19% | +0.10% |
| Total | 334 | +10.62 | +3.18% | 56.89% | 1.89 | 54.19% | +0.60% | +0.36% |

This is the best evidence so far that the system may contain useful signal.

Why this is promising:

- Profit is positive in all three tested seasons.
- Average CLV is positive overall.
- Positive CLV rate is above 50% in all three seasons.
- Longshots are restricted.
- Bet count is controlled.
- The test uses out-of-sample walk-forward structure.

Why it is still not proof:

- 334 bets is still a limited sample.
- ROI is modest.
- The 2024 average CLV is only +0.19%.
- There is no live odds ingestion yet.
- We have not added true xG, lineups, injuries, or player ratings.
- We have not checked drawdown or bankroll survival yet.

## Current Report Files

Most important reports:

| Report | Path |
| --- | --- |
| Constrained CLV walk-forward summary | `reports/club_odds_constrained_clv_walk_forward_summary.md` |
| 2022 constrained CLV report | `reports/club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md` |
| 2023 constrained CLV report | `reports/club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md` |
| 2024 constrained CLV report | `reports/club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md` |
| 2018 World Cup clean 3-way report | `reports/world_cup_2018_poisson_clean_3way_report.md` |
| 2022 World Cup clean 3-way report | `reports/world_cup_2022_poisson_clean_3way_report.md` |

## Testing Status

Latest automated test run:

- Command: `python -m unittest discover -s tests`
- Result: 39 tests passed

Test coverage includes:

- Market probability helpers.
- Elo feature engineering.
- International data processing.
- World Cup target filtering.
- Club data processing.
- Club rolling features.
- Calibration/blending.
- Odds helpers.
- Residual value model candidate generation and selection.

## xG Data Status

We researched xG sources.

Current conclusion:

- Reliable full historical xG likely requires paid data.
- Free xG/event data is useful for learning and prototyping, but not enough for our current big-five league betting backtest.

Candidate sources:

| Source | Status | Notes |
| --- | --- | --- |
| StatsBomb Open Data | Free but limited | High quality, but only selected competitions/seasons. Not enough for broad club betting backtests. |
| TheStatsAPI | Paid | Promising because it advertises xG, odds, match stats, and long historical coverage. Should be trialed before buying long-term. |
| Sportmonks xG | Paid add-on | Useful, but xG coverage may start too recently for deep historical training. |
| FootyStats API | Paid | Needs validation before relying on xG as a core model input. |

Recommendation:

Do not buy xG data yet. First add the CLV prediction model and drawdown/bankroll reports. If the positive signal survives, trial a paid xG source for one month.

## What Is Missing Before Real Use

The project still needs:

1. CLV prediction model.
   - Current residual value model predicts bet value/profit.
   - Next version should explicitly predict whether opening odds will beat closing odds.

2. Combined selection rule.
   - Select a bet only if:
     - residual value model approves,
     - CLV model approves,
     - odds are <= 4.00,
     - it is in the top 3 for that weekend,
     - bankroll rules permit it.

3. Drawdown and bankroll simulation.
   - Current tests use flat stakes.
   - We need maximum drawdown, losing streaks, variance, and bankroll survival.

4. Paper-trading workflow.
   - Store upcoming fixtures.
   - Store opening odds.
   - Produce recommendations.
   - Save predictions before matches.
   - Compare later with closing odds and final results.

5. Real odds ingestion.
   - The current system uses historical CSVs.
   - A real system needs an API connector and careful source mapping.

6. xG, player, lineup, and injury features.
   - These are likely the biggest missing football signals.

7. More markets.
   - We have market probability helpers, but not full odds/testing pipelines for every market in the bookmaker screenshots.

8. National-team paid odds validation.
   - World Cup/national-team betting should wait until the club system is stronger.

## Recommended Next Step

The next coding step should be:

Build a CLV prediction model and combine it with the residual value model.

The model should learn:

> Did this opening price beat the closing price?

This target is cleaner than predicting final match outcomes, because even good bets lose. If we can consistently predict positive CLV, we have stronger evidence of real market signal.

Suggested next experiment:

1. Use existing candidate rows.
2. Add a binary target:
   - `positive_clv = clv_decimal > 0`
3. Train a CLV classifier walk-forward.
4. Require double approval:
   - residual value edge positive,
   - CLV model predicts positive line movement.
5. Re-run 2022, 2023, 2024 constrained tests.
6. Compare:
   - profit,
   - ROI,
   - positive CLV rate,
   - average CLV,
   - bet count,
   - max drawdown.

## Final Readiness Answer

The model is not ready to start real betting.

It is ready to start structured paper trading and further model validation.

If we were to define readiness levels:

| Level | Status |
| --- | --- |
| Data pipeline | Ready for research |
| Market probability engine | Ready for research |
| Club historical odds backtesting | Working |
| Residual value filtering | Working |
| CLV reporting | Working |
| Walk-forward constrained signal | Promising |
| Live odds ingestion | Not built |
| xG/player/injury data | Not integrated |
| Bankroll/drawdown safety | Not built |
| Real-money betting readiness | Not ready |

Best practical next phase:

Paper trade only, with no real money, while we build the CLV classifier and bankroll/drawdown reporting.
