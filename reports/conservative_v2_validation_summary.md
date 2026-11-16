# Conservative V2 Validation

## Frozen Policy

- Use opening odds for decisions and settlement; use closing odds only for CLV.
- Anchor residual probabilities to bookmaker no-vig probabilities separately for `1X2` and `Over/Under 2.5`.
- Select the residual weight from `0%` through `50%` on validation log loss.
- Select a separate edge threshold for each market on the immediately preceding validation season.
- Require at least 20 validation bets and positive validation profit for an active market.
- Require at least `55%` estimated probability and odds no higher than `2.50`.
- Use flat one-unit stakes, at most one bet per market per weekend, one bet per match, and three bets per weekend overall.

## Rolling Holdouts

| test_season | selection_mode | status | model_weights | edge_thresholds | bets | profit | roi | hit_rate | avg_odds | positive_clv_rate | avg_decimal_clv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | conservative_v1 | active | residual only | all=2.0% | 105 | 12.420 | 0.118 | 0.695 | 1.605 | 0.543 | 0.011 |
| 2022 | conservative_v2 | active | 1x2=50%, over_under_2_5=50% | 1x2=0%, over_under_2_5=no-bet | 33 | 9.300 | 0.282 | 0.788 | 1.623 | 0.485 | 0.010 |
| 2023 | conservative_v1 | active | residual only | all=2.0% | 100 | -2.900 | -0.029 | 0.600 | 1.638 | 0.550 | 0.008 |
| 2023 | conservative_v2 | abstained | 1x2=20%, over_under_2_5=30% | 1x2=no-bet, over_under_2_5=no-bet | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2024 | conservative_v1 | active | residual only | all=1.0% | 102 | -12.290 | -0.120 | 0.539 | 1.603 | 0.559 | 0.004 |
| 2024 | conservative_v2 | abstained | 1x2=50%, over_under_2_5=50% | 1x2=no-bet, over_under_2_5=no-bet | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2025 | conservative_v1 | abstained | residual only | no-bet | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2025 | conservative_v2 | abstained | 1x2=50%, over_under_2_5=50% | 1x2=no-bet, over_under_2_5=no-bet | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Aggregate Comparison

| selection_mode | runs | active_runs | profitable_runs | bets | profit | roi | hit_rate | avg_odds | positive_clv_rate | avg_decimal_clv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative_v1 | 4 | 3 | 1 | 307 | -2.770 | -0.009 | 0.612 | 1.615 | 0.550 | 0.008 |
| conservative_v2 | 4 | 1 | 1 | 33 | 9.300 | 0.282 | 0.788 | 1.623 | 0.485 | 0.010 |

## Verdict

Conservative V1 placed 307 bets for -2.77 units and -0.90% ROI. Conservative V2 placed 33 bets for +9.30 units and +28.18% ROI.

V2 was active in only 1 of 4 test seasons. Its positive aggregate result therefore comes entirely from the 2022-2023 holdout and is not enough evidence of a repeatable betting edge. The later abstentions are useful capital protection, but the current policy is too sparse for practical deployment.

V2 remains a research candidate, not a real-money strategy. The next improvement should add genuinely independent information, especially historical xG and lineup availability, then repeat this exact rolling validation without tuning the test seasons.

## Reproducibility

Run each holdout with `scripts/backtest_conservative_v2.py`, then run `scripts/summarize_conservative_v2.py`.

Detailed reports, bet CSVs, JSON summaries, and the comparison CSV are stored under `reports/validation_runs_v2/` and intentionally ignored by Git. This file is the compact tracked result.
