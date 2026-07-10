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

## Development Setup

Create a local virtual environment and install the package with its dev tools
(`pytest`, `ruff`). A `Makefile` wraps the common commands:

```bash
make install        # python3 -m venv .venv && pip install -e ".[dev]"
make test           # run the test suite
make lint           # run ruff
make demo           # run the market-pricing demo (no data download)
```

The equivalent raw commands, if you prefer not to use `make`:

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

`pyproject.toml` sets `pythonpath = ["src", "."]` for pytest, so the tests run
without exporting `PYTHONPATH` manually. Continuous integration runs the linter
and the test suite on every push and pull request (see `.github/workflows/ci.yml`).

## Run The First Demo

From this folder (bash / macOS / Linux):

```bash
export PYTHONPATH=src
python scripts/build_dataset.py
python scripts/evaluate_holdout.py --holdout world_cup_2018
python scripts/evaluate_holdout.py --holdout world_cup_2018 --model poisson --learning-rate 0.03
python scripts/evaluate_holdout.py --holdout world_cup_2018 --model draw_poisson --learning-rate 0.03
python scripts/evaluate_holdout.py --holdout world_cup_2018 --model poisson --learning-rate 0.03 --target clean_3way
python examples/market_demo.py
python -m unittest discover -s tests
```

On Windows PowerShell, use `$env:PYTHONPATH = "src"` and `\` path separators.

## Free Club Odds Backtest

Before buying international odds, use Football-Data.co.uk club league CSVs to test the betting workflow:

```powershell
$env:PYTHONPATH = "src"
python scripts\build_club_odds_dataset.py
python scripts\backtest_club_odds.py --test-season 2024 --model poisson --odds-timing opening
python scripts\backtest_club_odds.py --test-season 2024 --model poisson --odds-timing opening --blend-with-bookmaker
python scripts\backtest_residual_value.py --test-season 2024 --base-model dc_poisson --odds-timing opening
python scripts\backtest_residual_value.py --test-season 2024 --base-model dc_poisson --odds-timing opening --max-odds 4.00 --max-bets-per-weekend 3
```

The first club backtest trains on earlier seasons from the big five leagues and tests on the 2024-2025 season. It compares our probabilities against no-vig bookmaker probabilities, then flat-stake backtests model value bets for `1X2` and `Over/Under 2.5` where the free odds columns are available. Opening odds are the default experiment because they leave room to evaluate whether the model can beat early prices.

Use `--blend-with-bookmaker` as a safety check. It learns, on the previous season only, how much to trust the model versus bookmaker no-vig probabilities before testing on the held-out season. If the learned alpha is `0`, the model has not earned the right to override the market yet.

The free club pipeline now adds pre-match rolling features for shots, shots on target, rest days, recent form, home form, and away form. It is ready to accept xG columns later, but Football-Data.co.uk does not provide xG in these CSVs, so xG is not faked.

The residual value backtest trains a second model on walk-forward bet candidates. It uses older seasons to learn when disagreement between our base model and opening odds has historically been useful, validates the edge threshold on the next season, then tests on the held-out season.

The constrained CLV version compares selected opening bets against closing odds. It also enforces the current betting discipline rules: no odds above `4.00` and at most 3 selected bets per club weekend, ranked by the model's selected probability.

## Current Research Status

The current best experimental path is:

- Dixon-Coles Poisson base model
- Residual value filter
- Opening odds
- Maximum odds: `4.00`
- Maximum club bets per weekend: `3`
- CLV reporting against closing odds

The CLV-prediction filter exists as an experiment, but it is not the active strategy because its 2022-2024 walk-forward result was worse than the plain residual value filter.
