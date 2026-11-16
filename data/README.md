# Data Folder

Use this folder to keep source data separate from model-ready data.

## Folders

- `raw/`: downloaded files exactly as received from the source.
- `processed/`: cleaned tables used by the model.
- `external/`: reference exports such as ratings tables.

The first build writes these files under `processed/`:

- `matches_processed.csv`
- `matches_with_elo_features.csv`
- `train.csv`
- `validation.csv`
- `test.csv`
- `dataset_metadata.json`
- `tournament_holdouts/world_cup_2018_train.csv`
- `tournament_holdouts/world_cup_2018_test.csv`
- `tournament_holdouts/world_cup_2022_train.csv`
- `tournament_holdouts/world_cup_2022_test.csv`
- `club_odds/club_matches.csv`
- `club_odds/club_matches_with_elo_features.csv`
- `club_odds/club_dataset_metadata.json`
- `lineups/international_starting_xi.csv`
- `lineups/international_lineup_features.csv`
- `lineups/player_matching_audit.csv`
- `lineups/lineup_features_metadata.json`

`external/world_cup_settlements.csv` contains curated 90-minute and qualification settlements for 2018 and 2022 World Cup knockout matches.

`external/world_cup_third_place_regulation.csv` contains the regulation-time scores of men's World Cup third-place matches through 2022. It is used only for a shrinkage-based match-context adjustment, and holdout runs filter it by date so later tournaments cannot leak into earlier forecasts. The match list is cross-checked against FIFA's historical third-place coverage and The Stats Zone's result list; the 1986 row explicitly separates the 2-2 regulation score from France's 4-2 extra-time result.

The club odds feature file includes pre-match rolling features derived only from previous matches:

- `*_rest_days_pre`
- `*_points_per_match_last5_pre`
- `*_goals_for_last5_pre`
- `*_goals_against_last5_pre`
- `*_shots_for_last5_pre`
- `*_shots_against_last5_pre`
- `*_sot_for_last5_pre`
- `*_sot_against_last5_pre`

It also keeps both opening and closing odds columns. Use opening odds for early-price experiments and closing odds for harder market-efficiency checks.

The lineup builder combines StatsBomb starting XIs with the latest Transfermarkt
valuation dated on or before kickoff. Run `python scripts/build_lineup_features.py`
to rebuild it. These features are valid only after official lineups are announced.

## Rules

- Do not overwrite raw files after download.
- Store source name and download date with every dataset.
- For each match row, only use information that would have been available before kickoff.
- Keep odds timestamps when available, because opening odds and closing odds answer different questions.
