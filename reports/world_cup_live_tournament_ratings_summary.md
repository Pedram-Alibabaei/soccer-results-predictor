# World Cup Live Tournament Ratings Experiment

## Why This Exists

The original World Cup holdout evaluation froze Elo ratings before the tournament began. That is strict and useful for pre-tournament forecasting, but it is not how we forecast late knockout matches during a live tournament. By the semifinal, group-stage and earlier knockout results are already known.

The new `--live-tournament-ratings` mode recomputes test features match by match:

1. Train the model before the tournament.
2. Predict the next tournament match using ratings available before that match.
3. After the match is complete, update Elo and rolling goal features.
4. Use those updated features for later tournament matches.

This avoids leaking a match's own result into its prediction while still using already-known tournament results.

## Clean 3-Way Holdout Results

Poisson goal model, `--target clean_3way`.

| Holdout | Mode | Accuracy | Log loss | Brier |
| --- | --- | ---: | ---: | ---: |
| World Cup 2018 | Strict pre-tournament | 0.562 | 0.951 | 0.563 |
| World Cup 2018 | Live tournament ratings | 0.594 | 0.947 | 0.561 |
| World Cup 2022 | Strict pre-tournament | 0.547 | 1.055 | 0.608 |
| World Cup 2022 | Live tournament ratings | 0.547 | 1.059 | 0.610 |

## Draw-Aware Check

Draw-aware Poisson was tested too, but it did not improve probability quality overall.

| Holdout | Mode | Accuracy | Log loss | Brier |
| --- | --- | ---: | ---: | ---: |
| World Cup 2018 | Strict pre-tournament | 0.547 | 0.962 | 0.570 |
| World Cup 2018 | Live tournament ratings | 0.578 | 0.960 | 0.569 |
| World Cup 2022 | Strict pre-tournament | 0.531 | 1.090 | 0.614 |
| World Cup 2022 | Live tournament ratings | 0.562 | 1.100 | 0.619 |

## Verdict

Live tournament ratings should be used for in-progress tournament forecasts, including semifinal and final predictions. It is not a dramatic model upgrade, but it better matches the information available at prediction time.

The plain Poisson goal model remains the current World Cup reference model. Draw-aware Poisson is still experimental because it improves draw top-picks slightly but hurts log loss.
