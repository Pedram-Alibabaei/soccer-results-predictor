# Market Roadmap

The uploaded screenshots show the long-term target market set. We can cover most of it by starting with a full-time score probability matrix.

## Phase 1: Full-Time Score Matrix

These markets can be derived directly from full-time score probabilities:

- `3-Way`: sum all home-win, draw, and away-win score probabilities.
- `Double chance`: combine the 3-way probabilities.
- `Over/Under`: sum score probabilities by total goals.
- `Both teams to score`: sum scores where both teams score at least once.
- `Correct score`: read directly from the score matrix.
- `Draw no bet`: home or away win with draw treated as a push.
- `European handicap`: adjust the final score and then recalculate 3-way probabilities.
- `Match Result & Over/Under`: combine match result with total-goals side.

## Phase 2: First-Half Markets

These need a separate first-half score probability matrix:

- `Who wins the 1st half?`
- `1st Half Over/Under`

The first-half model can reuse the same market logic, but it needs separate goal-rate estimates.

## Phase 3: Knockout Markets

These need extra modeling beyond 90 minutes:

- `Who will go through?`

This requires estimating:

- 90-minute result probabilities
- Extra-time goal probabilities
- Penalty shootout probabilities
- Team-specific penalty strength, goalkeeper quality, and historical tendencies

Important distinction: a knockout match can be a draw for `3-Way` betting after 90 minutes while still having a single team advance. The project should keep these as separate targets:

- `3-Way`: settled on 90-minute result.
- `Who will go through?`: settled on qualification after extra time and penalties.

The first results source does not reliably separate 90-minute, extra-time, and penalty outcomes for every knockout match, so strict 3-way evaluation currently uses the `clean_3way` target filter.

## Phase 4: Recommendation Layer

For every market, the recommendation layer should:

1. Convert bookmaker odds into implied probabilities.
2. Remove bookmaker margin where possible.
3. Compare model probability against market probability.
4. Calculate expected value.
5. Filter by minimum edge and bankroll rules.

The output should be a ranked list of suggested bets, not a promise that any single bet will win.
