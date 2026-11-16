# Confirmed-Lineup V4 (Position Strength): World Cup 2022

## Design

- Stable Poisson baseline: all historical international matches before each forecast date.
- Lineup correction training: World Cup 2018 and Euro 2020 only, using confirmed starting XIs.
- Player strength: latest Transfermarkt valuation on or before kickoff, log-scaled by position.
- Rotation: changes, goalkeeper replacement, defensive continuity, and attacking continuity.
- Hyperparameters selected on Euro 2020 after fitting the first correction on World Cup 2018.
- Final untouched test: all 64 World Cup 2022 matches, scored at 90 minutes.

## Selected Correction

- L2 regularization: 1.0
- Feature set: position_strength
- Correction strength: 1.00
- Confirmed-lineup training rows: 115
- Confirmed-lineup test rows: 64/64
- Maximum per-team goal-rate adjustment: +/-35% in log space.

## World Cup 2022 Results

| Model | Accuracy | Log loss | Brier |
| --- | ---: | ---: | ---: |
| Poisson baseline | 0.547 | 1.059 | 0.610 |
| Confirmed-lineup V4 | 0.500 | 1.105 | 0.629 |

Lower log loss and Brier are better; accuracy is secondary because betting needs calibrated probabilities.

## Euro 2020 Selection Grid

| L2 | Strength | Accuracy | Log loss | Brier |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 1.00 | 0.647 | 0.881 | 0.519 |
| 2.0 | 1.00 | 0.647 | 0.882 | 0.519 |
| 0.5 | 1.00 | 0.627 | 0.882 | 0.520 |
| 1.0 | 0.75 | 0.647 | 0.883 | 0.520 |
| 0.5 | 0.75 | 0.647 | 0.883 | 0.520 |
| 2.0 | 0.75 | 0.627 | 0.884 | 0.520 |
| 5.0 | 1.00 | 0.627 | 0.884 | 0.519 |
| 0.5 | 0.50 | 0.627 | 0.884 | 0.520 |

## Decision Rule

Promote V4 only if it improves probability quality on the untouched 2022 holdout. A correction strength of zero means validation rejected lineup adjustments and the stable baseline remains active.
