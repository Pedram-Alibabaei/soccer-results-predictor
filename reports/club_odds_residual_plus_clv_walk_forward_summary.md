# Residual + CLV Model Walk-Forward Summary

## Setup

- Base model: Dixon-Coles Poisson
- Odds timing: opening odds
- Markets: 1X2 and Over/Under 2.5
- Max odds: `4.00`
- Max club bets per weekend: `3`
- Test seasons: 2022, 2023, 2024
- CLV target: selected opening odds beat closing odds
- CLV model role: second approval filter after the residual value model

## Comparison

### Current Active Residual Value Model

| Test season | Bets | Profit | ROI | Hit rate | Avg odds | Positive CLV | Avg decimal CLV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 109 | 1.61 | 1.48% | 54.13% | 2.02 | 52.29% | 0.69% |
| 2023 | 114 | 4.93 | 4.32% | 57.02% | 1.84 | 56.14% | 0.92% |
| 2024 | 111 | 4.08 | 3.68% | 58.56% | 1.79 | 54.05% | 0.19% |
| Total | 334 | 10.62 | 3.18% | 56.89% | 1.89 | 54.19% | 0.60% |

### Experimental Residual + CLV Model

| Test season | Bets | Profit | ROI | Hit rate | Avg odds | Positive CLV | Avg decimal CLV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 104 | -2.49 | -2.39% | 50.96% | 2.10 | 50.96% | 0.74% |
| 2023 | 114 | 5.68 | 4.98% | 57.02% | 1.85 | 56.14% | 0.81% |
| 2024 | 72 | -10.79 | -14.99% | 33.33% | 2.93 | 61.11% | 0.77% |
| Total | 290 | -7.60 | -2.62% | 48.97% | 2.21 | 55.52% | 0.77% |

## Verdict

Do not promote the CLV model filter to the active strategy yet.

It improves the CLV-looking part of the profile, especially in 2024, but it damages the actual betting result. The key warning sign is 2024: positive CLV rose to 61.11%, yet ROI fell to -14.99% and average odds jumped to 2.93. That means the CLV filter is currently too willing to approve higher-priced outcomes where the closing line moves favorably but the bet still loses too often.

The current active model remains the plain residual value model with a CLV report/check, not the CLV-prediction filter.

## Keep As Experimental

Generated files:

- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2022_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2023_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_report.md`
- `club_odds_2024_dc_poisson_opening_maxodds_4p00_top3_residual_value_clvmodel_bets.csv`

## Next Improvement Ideas

- Use the CLV model as a diagnostic score rather than a hard approval filter.
- Add odds bands, for example separate thresholds for odds under 2.00, 2.00-3.00, and 3.00-4.00.
- Add a minimum model win probability rule before CLV can approve a bet.
- Evaluate whether CLV predicts future profitability over a larger club dataset, not only these six seasons.
