# Data Dictionary

The first processed dataset is `data/processed/matches_with_elo_features.csv`.

## Identity Columns

- `match_id`: Project-local match identifier after chronological sorting.
- `date`: Match date.
- `home_team`: Listed home team from the source.
- `away_team`: Listed away team from the source.
- `tournament`: Competition name.
- `city`: Venue city.
- `venue_country`: Venue country.
- `neutral`: Whether the source marks the match as neutral-site.

## Outcome Columns

- `home_goals`: Home-team goals.
- `away_goals`: Away-team goals.
- `result`: `1` for home win, `X` for draw, `2` for away win.
- `total_goals`: Home plus away goals.
- `goal_difference`: Home goals minus away goals.
- `both_teams_scored`: Whether both teams scored at least once.
- `is_clean_3way_result`: Whether the row is currently safe for strict 90-minute 3-way evaluation.
- `market_3way_home_goals`: Goals used for 90-minute 3-way settlement.
- `market_3way_away_goals`: Goals used for 90-minute 3-way settlement.
- `market_3way_result`: `1`, `X`, or `2` based on the 90-minute settlement score.
- `has_90_minute_settlement`: Whether a row has an explicit or source-safe 90-minute settlement.
- `qualification_result`: `1` or `2` for knockout advancement when available.
- `settlement_notes`: Human-readable note for curated settlement overrides.

For the initial source, knockout match goals may include extra time. Treat these outcome columns as good enough for early team-strength modeling, but not as final betting-settlement truth for 90-minute markets.

## World Cup Context Columns

- `is_world_cup`: Match is in the FIFA World Cup finals.
- `world_cup_year`: World Cup year, or `0` for non-World Cup rows.
- `world_cup_match_number`: Chronological match number within that World Cup.
- `world_cup_match_number_scaled`: Match number divided by tournament match count.
- `world_cup_stage`: Derived stage label.
- `is_world_cup_group_stage`: Derived group-stage flag.
- `is_world_cup_knockout_stage`: Derived knockout-stage flag.
- `is_world_cup_round_of_16`: Derived round-of-16 flag.
- `is_world_cup_quarter_final`: Derived quarter-final flag.
- `is_world_cup_semi_final`: Derived semi-final flag.
- `is_world_cup_third_place`: Derived third-place playoff flag.
- `is_world_cup_final`: Derived final flag.
- `world_cup_group_matchday`: Approximate group matchday, usually `1`, `2`, or `3`.
- `is_world_cup_final_group_match`: Approximate final group-match flag.

## Pre-Match Feature Columns

These are calculated before each match is used to update ratings.

- `home_elo_pre`: Home team's Elo before the match.
- `away_elo_pre`: Away team's Elo before the match.
- `elo_diff_pre`: Home Elo plus home-field adjustment minus away Elo.
- `home_attack_pre`: Home team's rolling goals-for estimate before the match.
- `away_attack_pre`: Away team's rolling goals-for estimate before the match.
- `home_defense_pre`: Home team's rolling goals-against estimate before the match.
- `away_defense_pre`: Away team's rolling goals-against estimate before the match.
- `home_matches_played_pre`: Home team's prior match count in this dataset.
- `away_matches_played_pre`: Away team's prior match count in this dataset.
- `home_expected_result_elo`: Elo-implied home expected result, where draw counts as half.

## Split Files

- `train.csv`: Older matches used for model fitting.
- `validation.csv`: Later matches used for model selection and tuning.
- `test.csv`: Newest matches held back for final evaluation.
- `dataset_metadata.json`: Source URL, row counts, date range, and split boundaries.

## Tournament Holdout Files

Tournament holdouts are written to `data/processed/tournament_holdouts/`.

- `world_cup_2018_train.csv`: All completed matches before the 2018 World Cup began.
- `world_cup_2018_test.csv`: 2018 World Cup finals matches only.
- `world_cup_2022_train.csv`: All completed matches before the 2022 World Cup began.
- `world_cup_2022_test.csv`: 2022 World Cup finals matches only.

The holdout test features use the pre-tournament Elo state. They do not update ratings with earlier rows from the same tournament.

## Confirmed-Lineup Files

Confirmed-lineup outputs are written to `data/processed/lineups/` and are valid
only after both official starting XIs are known.

- `international_starting_xi.csv`: One row per starter, including StatsBomb ID,
  position, Transfermarkt match audit, valuation date, and market value.
- `international_lineup_features.csv`: One row per match with rotation,
  continuity, and positional-strength features.
- `player_matching_audit.csv`: Unique cross-provider identity decisions and
  confidence scores.
- `lineup_features_metadata.json`: Sources, license, coverage, and leakage rule.

Important match-level fields:

- `has_confirmed_lineups`: Both teams have exactly eleven starters.
- `*_previous_lineup_available_pre`: A prior XI exists within the same tournament.
- `*_lineup_changes_pre`: Current starters absent from the previous XI.
- `*_goalkeeper_changed_pre`: The previous starting goalkeeper was replaced.
- `*_defensive_continuity_pre`: Current goalkeeper/defender overlap with the prior unit.
- `*_attacking_continuity_pre`: Current midfielder/attacker overlap with the prior unit.
- `*_goalkeeper_strength_pre`, `*_defense_strength_pre`,
  `*_midfield_strength_pre`, `*_attack_strength_pre`: Mean log market value for
  the named positional unit.
- `*_strength_change_pre`: Current positional or full-XI strength minus the
  previous tournament lineup's strength.
- `*_player_strength_coverage_pre`: Share of starters with a dated valuation.

Missing player values are imputed from the same tournament and positional unit,
while the original coverage remains visible. Valuations after kickoff are never
used.
