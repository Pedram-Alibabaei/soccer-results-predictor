# Confirmed-Lineup V4 (Combined): World Cup 2022

## Design

- Stable Poisson baseline: all historical international matches before each forecast date.
- Lineup correction training: World Cup 2018 and Euro 2020 only, using confirmed starting XIs.
- Player strength: latest Transfermarkt valuation on or before kickoff, log-scaled by position.
- Rotation: changes, goalkeeper replacement, defensive continuity, and attacking continuity.
- Hyperparameters selected on Euro 2020 after fitting the first correction on World Cup 2018.
- Final untouched test: all 64 World Cup 2022 matches, scored at 90 minutes.

## Selected Correction

- L2 regularization: 5.0
- Feature set: combined
- Correction strength: 1.00
- Confirmed-lineup training rows: 115
- Confirmed-lineup test rows: 64/64
- Maximum per-team goal-rate adjustment: +/-35% in log space.

## World Cup 2022 Results

| Model | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Poisson baseline | 0.547 | 1.059 | 0.610 |
| Confirmed-lineup V4 | 0.516 | 1.081 | 0.617 |

Lower log loss and Brier are better; accuracy is secondary because betting needs calibrated probabilities.

## Euro 2020 Selection Grid

| L2 | Strength | Accuracy | Log loss | Brier |
| ---: | ---: | ---: | ---: | ---: |
| 5.0 | 1.00 | 0.627 | 0.885 | 0.520 |
| 2.0 | 1.00 | 0.608 | 0.886 | 0.521 |
| 2.0 | 0.75 | 0.627 | 0.886 | 0.521 |
| 5.0 | 0.75 | 0.627 | 0.886 | 0.521 |
| 2.0 | 0.50 | 0.627 | 0.887 | 0.521 |
| 2.0 | 0.40 | 0.627 | 0.887 | 0.521 |
| 5.0 | 0.50 | 0.627 | 0.888 | 0.521 |
| 1.0 | 0.40 | 0.627 | 0.888 | 0.522 |

## Decision Rule

Promote V4 only if it improves probability quality on the untouched 2022 holdout. A correction strength of zero means validation rejected lineup adjustments and the stable baseline remains active.
