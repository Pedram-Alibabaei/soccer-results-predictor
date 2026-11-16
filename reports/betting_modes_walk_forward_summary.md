# Initial Big-Five Betting Modes Comparison

> Historical note: this was the first five-league comparison. The broader ten-league result in `expanded_betting_validation_summary.md` supersedes it and does not support real-money deployment.

## Purpose

This experiment separates bet selection into two user-facing modes while keeping the underlying Dixon-Coles and residual value models unchanged.

- `value`: rank positive-EV candidates by residual expected edge, allow decimal odds up to `4.00`.
- `conservative`: require at least `55%` predicted win probability, cap decimal odds at `2.50`, and rank by win probability.

Both modes use opening odds, flat one-unit stakes, at most one bet per match, and at most three bets per club weekend. A residual edge threshold is selected on the immediately preceding validation season. If no threshold is profitable with at least 50 validation bets, the mode abstains for the test season.

## Held-Out Results

| Test season | Mode | Bets | Profit | ROI | Hit rate | Avg odds | Positive CLV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2022 | Value | 109 | +6.21 | +5.70% | 45.87% | 2.58 | 57.80% |
| 2022 | Conservative | 61 | +9.23 | +15.13% | 68.85% | 1.71 | 49.18% |
| 2023 | Value | 0 | Abstained | N/A | N/A | N/A | N/A |
| 2023 | Conservative | 94 | +3.11 | +3.31% | 64.89% | 1.61 | 54.26% |
| 2024 | Value | 0 | Abstained | N/A | N/A | N/A | N/A |
| 2024 | Conservative | 0 | Abstained | N/A | N/A | N/A | N/A |

## Combined Active Bets

| Mode | Bets | Profit | ROI | Hit rate | Avg odds | Positive CLV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Value | 109 | +6.21 | +5.70% | 45.87% | 2.58 | 57.80% |
| Conservative | 155 | +12.34 | +7.96% | 66.45% | 1.65 | 52.26% |

ROI is calculated only over units actually staked. An abstained season is reported as `N/A`, not as a zero-percent return.

## Interpretation

The modes are behaving differently in the intended way. Value mode accepts lower-probability outcomes when the price creates a larger estimated edge. Conservative mode produces a much higher hit rate and lower average odds while still requiring positive expected value.

The conservative mode is the stronger result in this first comparison: more profit, higher ROI, and a substantially higher hit rate. It should not yet be treated as proven. It placed only 155 bets, its aggregate positive-CLV rate is a modest `52.26%`, and both modes abstained in 2024 because their 2023 validation results did not support betting.

Both modes are worth retaining for the eventual app. The interface should clearly show predicted probability, fair odds, bookmaker odds, estimated edge, and the rule that qualified each bet. The next research step is to test these fixed mode definitions over more held-out seasons and leagues before changing their thresholds.
