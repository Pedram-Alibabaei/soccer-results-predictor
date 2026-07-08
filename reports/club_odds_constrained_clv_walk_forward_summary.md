# Constrained CLV Walk-Forward Summary

## Setup

- Base model: Dixon-Coles Poisson
- Odds timing: opening odds
- CLV comparison: selected opening odds vs closing odds
- Max odds: 4.00
- Max club bets per weekend: 3
- Residual value model training: walk-forward out-of-sample candidates
- Early-history setting: `--min-base-train-seasons 1`

## Season Results

| Test season | Bets | Profit | ROI | Hit rate | Avg odds | Positive CLV | Avg decimal CLV | Avg no-vig probability CLV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | 109 | 1.61 | 1.48% | 54.13% | 2.02 | 52.29% | 0.69% | 0.44% |
| 2023 | 114 | 4.93 | 4.32% | 57.02% | 1.84 | 56.14% | 0.92% | 0.55% |
| 2024 | 111 | 4.08 | 3.68% | 58.56% | 1.79 | 54.05% | 0.19% | 0.10% |
| Total | 334 | 10.62 | 3.18% | 56.89% | 1.89 | 54.19% | 0.60% | 0.36% |

## Notes

2021 was not included because the free dataset begins in 2019, leaving too little prior history to train a residual value model and validate it honestly.

This is a promising signal, not proof. Profit is positive in all three tested seasons and average CLV is positive overall, but the sample is still small and the average edge is modest.
