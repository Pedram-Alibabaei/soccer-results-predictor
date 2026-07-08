# Reports Folder

This folder now keeps only the current reference reports.

## Current Active Club Model

The current most promising betting path is:

- Dixon-Coles Poisson base model
- Residual value filter
- Opening odds
- Maximum odds: `4.00`
- Maximum club bets per weekend: `3`
- CLV check against closing odds

Current active files:

- `club_odds_constrained_clv_walk_forward_summary.md`
- `club_odds_residual_plus_clv_walk_forward_summary.md`
- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md`
- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_bets.csv`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_bets.csv`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_bets.csv`

## Current Experimental Club Model

The CLV-prediction filter was tested as a second approval layer. It is not the active strategy because it lost money overall in the 2022-2024 walk-forward test, despite improving average CLV.

- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`

## Current World Cup Reference

The World Cup reports are kept only as national-team probability references. They are not betting-ready because historical World Cup odds are not yet integrated.

- `world_cup_2018_poisson_clean_3way_report.md`
- `world_cup_2018_poisson_clean_3way_predictions.csv`
- `world_cup_2022_poisson_clean_3way_report.md`
- `world_cup_2022_poisson_clean_3way_predictions.csv`

## Archived Experiments

Older model experiments and exploratory outputs are in:

- `archive_outdated_models/`
