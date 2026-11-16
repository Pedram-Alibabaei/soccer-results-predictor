# Soccer Results Predictor

This is a research project for football match probability modeling, odds analysis, and walk-forward backtesting. It started with World Cup 2026 betting-market ideas, then expanded into a club-league laboratory using historical match and odds data.

The goal is not to predict every winner. The goal is to estimate fair probabilities, compare them with bookmaker odds, and study whether a model can identify positive expected-value situations under realistic historical testing.

This repository is for educational and research use. It is not financial advice, gambling advice, or a ready-to-use betting system.

## First Target

The first version will focus on full-time soccer markets:

- 3-way result: `1`, `X`, `2`
- Double chance: `1X`, `12`, `X2`
- Over/under goals
- Both teams to score
- Draw no bet
- Correct score
- European handicap
- Match result plus over/under combinations

Later versions can add first-half markets and knockout-specific markets such as `Who will go through?`.

## Core Idea

Most of the target markets can be derived from a score probability table:

```text
Portugal 0-0 Croatia: 6.2%
Portugal 1-0 Croatia: 10.8%
Portugal 1-1 Croatia: 9.7%
Portugal 2-1 Croatia: 8.5%
...
```

Once we estimate that table, we can price many markets from the same source of truth.

## MVP Roadmap

1. Build a score-probability engine.
2. Build odds helpers: implied probability, bookmaker margin removal, expected value.
3. Collect historical match data and team ratings.
4. Build a first baseline model using Elo and attack/defense strength.
5. Backtest historical betting decisions.
6. Add player ratings, injuries, lineups, travel, rest days, venue, and weather.

## Data Plan

Initial useful sources:

- Football-Data.co.uk for historical match results and odds.
- World Football Elo Ratings and/or a project-owned Elo calculation for national-team strength.
- FIFA/Coca-Cola Men's World Ranking as a secondary national-team rating reference.
- EA official ratings, FIFA Index, SoFIFA, and Kaggle datasets for FIFA/EA-style player and team ratings.
- Later: injury, lineup, weather, referee, and venue feeds.

EA/FIFA-style ratings should be treated as supporting features, not as absolute truth. They are useful because they compress scouting knowledge into structured numbers.

For training and testing, the project should use chronological splits rather than random splits. Soccer betting is a time-series problem, so every test row must happen after every training row used to build that model.

## Project Structure

```text
data/
  raw/          downloaded source data, unchanged
  processed/    cleaned model-ready tables
  external/     third-party reference exports
docs/           design notes and market roadmap
examples/       small runnable examples
models/         trained model artifacts
notebooks/      research notebooks
reports/        backtest reports and analysis
src/            project package
tests/          automated tests
```

Generated datasets, trained model files, cache files, and bulky CSV backtest outputs are intentionally ignored by Git. The code and documentation are versioned; large or source-specific data should be rebuilt locally with the scripts below.

## Run The First Demo

From this folder:

```powershell
$env:PYTHONPATH = "src"
python scripts\build_dataset.py
python scripts\evaluate_holdout.py --holdout world_cup_2018
python scripts\evaluate_holdout.py --holdout world_cup_2018 --model poisson --learning-rate 0.03
python scripts\evaluate_holdout.py --holdout world_cup_2018 --model draw_poisson --learning-rate 0.03
python scripts\evaluate_holdout.py --holdout world_cup_2018 --model poisson --learning-rate 0.03 --target clean_3way
python scripts\evaluate_holdout.py --holdout world_cup_2018 --model poisson --learning-rate 0.03 --target clean_3way --live-tournament-ratings
python scripts\evaluate_goal_risk_v3.py
python scripts\build_lineup_features.py
python scripts\evaluate_lineup_v4.py --feature-set combined
python scripts\evaluate_lineup_v4.py --feature-set rotation
python scripts\evaluate_lineup_v4.py --feature-set position_strength
python examples\market_demo.py
python -m unittest discover -s tests
```

Use `--live-tournament-ratings` when evaluating an in-progress tournament forecast. The strict holdout mode freezes team ratings before the tournament starts; live mode updates Elo and rolling goal features after each completed tournament match before forecasting the next one.

`evaluate_goal_risk_v3.py` runs leakage-safe 2018 and 2022 World Cup holdouts for the experimental contextual goal-risk model. V3 removes the unstable third-place one-hot coefficient from the base fit, applies a shrinkage-based third-place multiplier learned only from earlier tournaments, and uses a Negative Binomial score distribution to expose `5+`, `6+`, and `10+` goal tail risk. It is a diagnostic experiment, not an approved betting policy.

## Confirmed-Lineup V4

The V4 data pipeline implements the two pre-match features that depend on the
official starting XIs:

- Lineup rotation: player changes, goalkeeper replacement, defensive continuity,
  and attacking continuity versus the team's previous tournament match.
- Player strength by position: goalkeeper, defense, midfield, and attack using
  the latest Transfermarkt valuation dated on or before kickoff.

StatsBomb Open Data supplies all 179 starting XIs for World Cup 2018, Euro 2020,
and World Cup 2022. The CC0 `transfermarkt-datasets` database supplies historical
valuations. The current identity matcher covers 95.8% of starter records and
90.0% have a usable dated valuation; missing values are conservatively imputed
from the tournament and position while coverage remains an explicit feature.

V4 is a regularized correction on top of the stable Poisson baseline. It is not
part of early forecasts because starting XIs are not known then. The correction
is trained on World Cup 2018, selected on Euro 2020, refit on both, and tested on
World Cup 2022. None of the tested variants beat the baseline on 2022 log loss:

- Baseline: `1.059`
- Rotation only: `1.075`
- Rotation plus positional strength: `1.081`
- Positional strength only: `1.105`

The features remain available as confirmed-lineup diagnostics, but V4 is not
promoted to the active probability or betting policy. More historical lineup
training data is required before another promotion test.

## Free Club Odds Backtest

Before buying international odds, use Football-Data.co.uk club league CSVs to test the betting workflow:

```powershell
$env:PYTHONPATH = "src"
python scripts\build_club_odds_dataset.py
python scripts\audit_club_data_coverage.py
python scripts\backtest_club_odds.py --test-season 2024 --model poisson --odds-timing opening
python scripts\backtest_club_odds.py --test-season 2024 --model poisson --odds-timing opening --blend-with-bookmaker
python scripts\backtest_residual_value.py --test-season 2024 --base-model dc_poisson --odds-timing opening
python scripts\backtest_residual_value.py --test-season 2024 --base-model dc_poisson --odds-timing opening --max-odds 4.00 --max-bets-per-weekend 3
python scripts\backtest_residual_value.py --test-season 2024 --selection-mode value
python scripts\backtest_residual_value.py --test-season 2024 --selection-mode conservative
python scripts\backtest_residual_value.py --test-season 2023 --selection-mode conservative --holdout-league N1 --report-subdir validation_runs
python scripts\summarize_betting_validation.py
python scripts\backtest_conservative_v2.py --test-season 2025
python scripts\summarize_conservative_v2.py
```

The first club backtest trains on earlier seasons from the big five leagues and tests on the 2024-2025 season. It compares our probabilities against no-vig bookmaker probabilities, then flat-stake backtests model value bets for `1X2` and `Over/Under 2.5` where the free odds columns are available. Opening odds are the default experiment because they leave room to evaluate whether the model can beat early prices.

Use `--blend-with-bookmaker` as a safety check. It learns, on the previous season only, how much to trust the model versus bookmaker no-vig probabilities before testing on the held-out season. If the learned alpha is `0`, the model has not earned the right to override the market yet.

The free club pipeline now adds pre-match rolling features for shots, shots on target, rest days, recent form, home form, and away form. It is ready to accept xG columns later, but Football-Data.co.uk does not provide xG in these CSVs, so xG is not faked.

The residual value backtest trains a second model on walk-forward bet candidates. It uses older seasons to learn when disagreement between our base model and opening odds has historically been useful, validates the edge threshold on the next season, then tests on the held-out season.

The constrained CLV version compares selected opening bets against closing odds. It also enforces the current betting discipline rules: no odds above `4.00` and at most 3 selected bets per club weekend, ranked by the model's selected probability.

Two selection modes are available for the same scored candidates:

- `value` requires positive expected value and ranks eligible bets by estimated edge, with the global `4.00` odds cap.
- `conservative` also requires positive expected value, but defaults to at least `55%` predicted win probability, odds no higher than `2.50`, and ranking by win probability.

Both modes choose their residual edge threshold from the validation season before they are measured on the held-out test season. This keeps their ROI comparison out of sample.

Conservative V2 is a stricter research policy. It shrinks residual-model probabilities toward the bookmaker's no-vig opening probabilities, learns separate residual weights and edge thresholds for `1X2` and `Over/Under 2.5`, and permits at most one selection from each market per weekend. All weights and thresholds are chosen on the preceding validation season before the test season is scored.

## Current Research Status

The current research architecture is:

- Dixon-Coles Poisson base model
- Residual value filter
- Opening odds
- Maximum odds: `4.00`
- Maximum club bets per weekend: `3`
- CLV reporting against closing odds

The expanded free dataset contains 24,165 matches from ten leagues across seven seasons (2019-2020 through 2025-2026). All 70 league-seasons pass the current 95% opening-odds, closing-odds, Over/Under, and shot-stat coverage gate.

Broader walk-forward validation does not currently support real-money deployment:

- Conservative mode: 307 pooled bets, `-2.77` units, `-0.90%` ROI.
- Value mode: 119 pooled bets, `-18.69` units, `-15.71%` ROI.
- Both modes abstained in the fresh 2025-2026 holdout after failing the 2024-2025 validation gate.

The bookmaker-anchored Conservative V2 experiment reduced activity sharply:

- Conservative V2: 33 pooled bets, `+9.30` units, `+28.18%` ROI, and a `78.79%` hit rate.
- The result came entirely from the 2022-2023 holdout; V2 abstained in the following three test seasons.
- Only `48.48%` of its bets had positive CLV, although average decimal CLV was slightly positive.

That sample is too small and concentrated to establish a repeatable edge. V2 remains the most promising conservative research candidate, but it is not ready for real-money use. See `reports/conservative_v2_validation_summary.md` for the frozen policy and complete comparison.

The leave-one-league-out diagnostic favors conservative selection, but its samples are too small and its pooled production-style ROI remains negative. The CLV-prediction filter also remains inactive because its earlier walk-forward result lost money.
