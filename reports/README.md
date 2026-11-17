# Reports Folder

This folder now keeps only the current reference reports.

Tracked Markdown files are compact historical research artifacts, not results
automatically reproduced by cloning the repository. Detailed CSV/JSON outputs and
archived experiments listed below are local, generated files and are not committed
to Git. Their names describe expected artifacts rather than downloadable files.
The [data guide](../data/README.md) explains the untracked curated inputs needed
for some international experiments. Rebuilding against changed upstream datasets
can change coverage and metrics.

## Current Club Research

The current model architecture is:

- Dixon-Coles Poisson base model
- Residual value filter
- Opening odds
- Maximum odds: `4.00`
- Maximum club bets per weekend: `3`
- CLV check against closing odds

Current club reference artifacts (Markdown tracked; CSV generated locally):

- `club_data_coverage_report.md`
- `expanded_betting_validation_summary.md`
- `conservative_v2_validation_summary.md`
- `betting_modes_walk_forward_summary.md`
- `club_odds_constrained_clv_walk_forward_summary.md`
- `club_odds_residual_plus_clv_walk_forward_summary.md`
- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md`
- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_bets.csv`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_bets.csv`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_report.md`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_bets.csv`

The expanded report supersedes the initial five-league mode comparison as the current verdict. Neither mode is approved for real-money deployment: pooled conservative ROI is slightly negative and pooled value ROI is clearly negative. Detailed expanded run artifacts are kept locally under the ignored `validation_runs/` folder.

`conservative_v2_validation_summary.md` compares the original conservative policy with the bookmaker-anchored V2 policy. V2 returned +9.30 units from 33 bets in the 2022-2023 holdout, then abstained in the next three test seasons. It is the leading conservative research candidate, but the result is too sparse to approve for deployment. Detailed V2 artifacts are kept locally under the ignored `validation_runs_v2/` folder.

## Current Experimental Club Model

The CLV-prediction filter was tested as a second approval layer. It is not the active strategy because it lost money overall in the 2022-2024 walk-forward test, despite improving average CLV.

- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`

## Current World Cup Reference

The World Cup reports are kept only as national-team probability references. They are not betting-ready because historical World Cup odds are not yet integrated. Reproducing the corrected full-tournament results requires `data/external/world_cup_settlements.csv`; contextual V3 also requires `data/external/world_cup_third_place_regulation.csv`. Neither curated input is included in Git. See the [research guide](../docs/research-guide.md) before running these experiments.

- `world_cup_2018_poisson_clean_3way_report.md`
- `world_cup_2018_poisson_clean_3way_live_report.md`
- `world_cup_2018_poisson_clean_3way_predictions.csv`
- `world_cup_2022_poisson_clean_3way_report.md`
- `world_cup_2022_poisson_clean_3way_live_report.md`
- `world_cup_2022_poisson_clean_3way_predictions.csv`
- `world_cup_live_tournament_ratings_summary.md`
- `contextual_goal_risk_v3_holdouts.md`
- `contextual_goal_risk_v3_predictions.csv`
- `lineup_features_coverage_report.md`
- `lineup_v4_combined_world_cup_2022_summary.md`
- `lineup_v4_rotation_world_cup_2022_summary.md`
- `lineup_v4_position_strength_world_cup_2022_summary.md`

The contextual V3 report compares the independent Poisson score distribution with a shrinkage-based third-place adjustment and a Negative Binomial tail. Its first 2018/2022 holdouts are mixed, so V3 remains a risk-reporting experiment and does not replace the current model.

The confirmed-lineup V4 reports compare lineup rotation, positional-strength
change, and the combined correction. All are trained chronologically on World
Cup 2018, selected on Euro 2020, and tested on World Cup 2022. Rotation-only was
the least harmful variant, but no V4 variant improved the baseline's 2022 log
loss, so these features remain advisory rather than active.

## Archived Experiments

Older model experiments and exploratory outputs are in:

- `archive_outdated_models/`
