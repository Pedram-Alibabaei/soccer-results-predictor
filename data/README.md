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

`external/world_cup_settlements.csv` contains curated 90-minute and qualification settlements for 2018 and 2022 World Cup knockout matches.

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

## Rules

- Do not overwrite raw files after download.
- Store source name and download date with every dataset.
- For each match row, only use information that would have been available before kickoff.
- Keep odds timestamps when available, because opening odds and closing odds answer different questions.
