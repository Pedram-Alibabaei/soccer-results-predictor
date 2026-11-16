# Expanded Betting Validation

## Fixed Policies

- Value: opening odds, positive residual EV, odds at most `4.00`, rank by edge.
- Conservative: opening odds, positive residual EV, probability at least `55%`, odds at most `2.50`, rank by probability.
- Both: flat one-unit stakes, one bet per match, at most three bets per club weekend.
- Edge threshold: selected on the immediately preceding validation season with at least 50 validation bets.

## Pooled Rolling Holdouts

These runs pool all ten leagues and are the closest current approximation to deploying one mode across the full club schedule.

| test_season | selection_mode | status | chosen_edge_threshold | bets | profit | roi | hit_rate | avg_odds | positive_clv_rate | avg_decimal_clv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | conservative | active | 0.020 | 105 | 12.420 | 0.118 | 0.695 | 1.605 | 0.543 | 0.011 |
| 2022 | value | abstained |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2023 | conservative | active | 0.020 | 100 | -2.900 | -0.029 | 0.600 | 1.638 | 0.550 | 0.008 |
| 2023 | value | active | 0.000 | 119 | -18.690 | -0.157 | 0.345 | 2.589 | 0.529 | 0.006 |
| 2024 | conservative | active | 0.010 | 102 | -12.290 | -0.120 | 0.539 | 1.603 | 0.559 | 0.004 |
| 2024 | value | abstained |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2025 | conservative | abstained |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2025 | value | abstained |  | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### Pooled Aggregate

| selection_mode | runs | active_runs | profitable_runs | bets | profit | roi | hit_rate | avg_odds | positive_clv_rate | avg_decimal_clv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | 4 | 3 | 1 | 307 | -2.770 | -0.009 | 0.612 | 1.615 | 0.550 | 0.008 |
| value | 4 | 1 | 0 | 119 | -18.690 | -0.157 | 0.345 | 2.589 | 0.529 | 0.006 |

## Leave-One-League-Out Diagnostic

For each row, model development and threshold selection exclude the named league. The final test contains only that league. These rows diagnose cross-league generalization; their profits must not be combined as a production portfolio because each run independently applies the three-bets-per-weekend cap.

| test_season | holdout_league | selection_mode | status | chosen_edge_threshold | bets | profit | roi | hit_rate | positive_clv_rate | avg_decimal_clv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | B1 | conservative | active | 0.020 | 5 | 1.960 | 0.392 | 0.800 | 0.600 | 0.028 |
| 2023 | B1 | value | active | 0.000 | 59 | -22.180 | -0.376 | 0.288 | 0.593 | 0.021 |
| 2023 | D1 | conservative | active | 0.030 | 16 | -2.240 | -0.140 | 0.562 | 0.500 | -0.001 |
| 2023 | D1 | value | active | 0.000 | 94 | -17.720 | -0.189 | 0.362 | 0.447 | -0.009 |
| 2023 | E0 | conservative | active | 0.000 | 53 | 4.090 | 0.077 | 0.698 | 0.509 | -0.003 |
| 2023 | E0 | value | active | 0.050 | 30 | -8.330 | -0.278 | 0.333 | 0.567 | 0.008 |
| 2023 | E1 | conservative | active | 0.030 | 13 | 3.120 | 0.240 | 0.692 | 0.615 | 0.015 |
| 2023 | E1 | value | active | 0.000 | 100 | -13.170 | -0.132 | 0.340 | 0.540 | 0.004 |
| 2023 | F1 | conservative | active | 0.010 | 17 | -3.170 | -0.186 | 0.471 | 0.706 | 0.042 |
| 2023 | F1 | value | active | 0.000 | 82 | 5.390 | 0.066 | 0.451 | 0.524 | 0.010 |
| 2023 | I1 | conservative | active | 0.020 | 15 | 2.200 | 0.147 | 0.667 | 0.467 | -0.013 |
| 2023 | I1 | value | active | 0.000 | 94 | -23.380 | -0.249 | 0.340 | 0.447 | -0.003 |
| 2023 | N1 | conservative | active | 0.030 | 7 | 2.770 | 0.396 | 0.857 | 0.143 | -0.027 |
| 2023 | N1 | value | active | 0.000 | 65 | -4.490 | -0.069 | 0.446 | 0.477 | -0.001 |
| 2023 | P1 | conservative | active | 0.000 | 11 | 0.560 | 0.051 | 0.636 | 0.727 | 0.032 |
| 2023 | P1 | value | active | 0.000 | 60 | 8.930 | 0.149 | 0.433 | 0.650 | 0.030 |
| 2023 | SC0 | conservative | active | 0.030 | 6 | -1.730 | -0.288 | 0.500 | 0.333 | -0.025 |
| 2023 | SC0 | value | active | 0.000 | 62 | -11.640 | -0.188 | 0.387 | 0.435 | -0.017 |
| 2023 | SP1 | conservative | active | 0.030 | 12 | 4.940 | 0.412 | 0.833 | 0.500 | 0.030 |
| 2023 | SP1 | value | active | 0.080 | 21 | -3.940 | -0.188 | 0.333 | 0.571 | 0.017 |

### Holdout Aggregate Diagnostic

| selection_mode | runs | active_runs | profitable_runs | bets | profit | roi | hit_rate | avg_odds | positive_clv_rate | avg_decimal_clv |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| conservative | 10 | 10 | 7 | 155 | 12.500 | 0.081 | 0.665 | 1.642 | 0.529 | 0.007 |
| value | 10 | 10 | 2 | 667 | -90.530 | -0.136 | 0.375 | 2.550 | 0.513 | 0.004 |

## Current Verdict

The pooled conservative mode placed 307 bets for -2.77 units and -0.90% ROI. The pooled value mode placed 119 bets for -18.69 units and -15.71% ROI.

Both modes abstained in the fresh 2025-2026 holdout because the preceding validation season did not produce a sufficiently supported profitable threshold.

The broader pooled evidence does not support deploying either mode for real-money betting. Conservative selection generalizes better than value selection in the league-holdout diagnostic, but its pooled ROI is still negative and the holdout samples are often small. Positive CLV alongside negative ROI is useful research evidence, not proof of profitability.

## Reproducibility

Machine-readable results: `C:\Users\Asus ProArt\Desktop\Soccer Results Predictor\reports\validation_runs\expanded_betting_validation_results.csv`

Detailed run reports, selected-bet CSVs, and JSON summaries are stored under `reports/validation_runs/` and intentionally ignored by Git. This report is the compact tracked artifact.
