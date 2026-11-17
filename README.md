# Soccer Results Predictor

A Python research project for estimating football match probabilities and
evaluating them on chronological holdouts. It combines historical match data,
Elo features and Poisson goal models, with experiments in lineup information and
odds-based backtesting.

The emphasis is on data preparation, reproducible evaluation and documenting
where a model fails. This is research software, not a deployed betting service
or evidence of a reliable betting edge.

**Stack:** Python, pandas, NumPy, DuckDB, scikit-learn, unittest.

## Pipeline

```text
Historical match data -> cleaning -> pre-match features -> probability models
                                                             |
                                   chronological holdouts / walk-forward tests
                                                             |
                                      log loss, Brier scores and research reports
```

## Project Ownership

This is Pedram Alibabaei's personal, AI-assisted research project. Its scope
includes historical data preparation, model experiments and evaluation scripts.
AI tools supported implementation and debugging. Automated tests and historical
evaluation are used to check behaviour, with the limitations below.

## Quick Start

Use **Python 3.11 or 3.12**. Some pipeline modules use `datetime.UTC`, which is not
available in Python 3.10. Run the following from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe examples\market_demo.py
```

After installing dependencies, this example needs no datasets, API keys or data
downloads. It uses fixed illustrative goal rates for Portugal and Croatia and
prints score-matrix probabilities and derived markets. It is a mathematical
example, not a trained forecast for an actual fixture.

Expected first output:

```text
Portugal vs Croatia
Matrix total probability: 1.000000
```

On macOS/Linux, use `.venv/bin/python` and forward slashes in place of the Windows
environment paths.

## Testing

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Tests cover market probabilities, Elo updates, modelling, data transformations and
evaluation helpers. Passing tests does not prove that an entire historical
research run is free from leakage or that a strategy is profitable.

## Implemented and Experimental Features

| Area | Current scope |
| --- | --- |
| Data | International results, shootouts, club results/odds and rolling features |
| Models | Elo features, softmax and Poisson-based probability models |
| Evaluation | Chronological holdouts, walk-forward tests, log loss and Brier scores |
| Market helpers | Three-way result, totals, both teams to score and related score-derived markets |
| Experiments | Lineup corrections, contextual goal risk and odds-based selection policies |

Additional historical inputs and stronger independent predictive information are
research directions, not implemented performance improvements. No live trading
or sportsbook execution is provided.

## Recorded Results and Limitations

These figures summarise **tracked historical reports**. They are not newly
reproduced results for the current upstream datasets:

- Expanded club holdouts: conservative selection placed 307 bets for -0.90% ROI;
  value selection placed 119 bets for -15.71% ROI. See the
  [expanded validation report](reports/expanded_betting_validation_summary.md).
- Conservative V2 recorded +28.18% ROI on only 33 bets, all from one holdout. It
  abstained in the next three test seasons. This is too small and concentrated
  to demonstrate a repeatable edge. See the
  [V2 report](reports/conservative_v2_validation_summary.md).
- The combined lineup correction had World Cup 2022 log loss of 1.081 versus
  1.059 for the Poisson baseline; lower is better. The experiment did not improve
  this holdout. See the [lineup report](reports/lineup_v4_combined_world_cup_2022_summary.md).

The main limitations are limited independent holdouts, upstream data changes,
incomplete point-in-time coverage and dependence on curated inputs for some
experiments. These reports do not establish production readiness.

## Optional Data and Research Workflows

The quick-start above is self-contained. Larger studies require network access,
third-party datasets and longer-running scripts; see the
[research guide](docs/research-guide.md).

Two curated inputs used by the recorded international experiments are **not
included in Git and have no supplied reconstruction script**:

- `data/external/world_cup_settlements.csv`: corrected 90-minute knockout scores.
  The builder treats it as optional, so a fresh clone can run a basic source-label
  study but need not reproduce the corrected full-tournament reports.
- `data/external/world_cup_third_place_regulation.csv`: required by the optional
  contextual goal-risk V3 experiment.

Do not run those advanced experiments expecting identical results until the
required inputs, source versions and settings have been supplied and checked.
Raw data, trained artifacts and detailed result CSVs are deliberately not tracked.

## Documentation

- [Research workflows and model notes](docs/research-guide.md)
- [Dataset builders and required inputs](data/README.md)
- [Current report index and interpretation](reports/README.md)
- [Data dictionary](docs/data-dictionary.md)
- [Data-source plan](docs/data-plan.md)

Check the original providers' terms before redistributing datasets. No software
licence has been added here. This project is educational research, not financial
or gambling advice.
