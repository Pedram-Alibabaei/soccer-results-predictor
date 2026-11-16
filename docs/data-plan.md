# Data Plan

The project needs two separate ideas of data:

1. Model features and outcomes, used to estimate match probabilities.
2. Bookmaker odds, used to decide whether a bet has positive expected value.

## First Training Dataset

Start with men's international matches:

- Match date
- Home team and away team
- Neutral-site flag
- Tournament
- City and country
- Full-time goals
- Result label: `1`, `X`, `2`

The first candidate source is Mart Jürisoo's `international_results` dataset on GitHub. It provides a broad men's international match-results table and is appropriate for building the first national-team model.

Important caveat: this source is useful for strength modeling, but it is not a perfect betting-settlement source for every market. Some knockout match scores reflect the final score including extra time rather than the 90-minute score used for most `3-Way` bets. Before we backtest actual bets, we should add a source that clearly separates 90-minute, extra-time, and penalty outcomes.

For now, the pipeline adds `is_clean_3way_result`. World Cup knockout rows are marked false because some source scores include extra time. These rows can still help broader strength modeling, but they should be filtered out for strict 90-minute `3-Way` evaluation until a true 90-minute settlement source is added.

For World Cup 2018 and 2022, `data/external/world_cup_settlements.csv` adds curated 90-minute scores and qualification outcomes for knockout matches. The pipeline also downloads `shootouts.csv` from the same international-results project to derive qualification winners where the source score remains tied.

Qualification evaluation has a source-order caveat: in these rows, the listed first team advances very often, so a naive "first listed team advances" baseline can look strong. A real betting-market version should evaluate team labels as displayed by the bookmaker and should compare against bookmaker qualification odds.

## First Ratings Dataset

Use one of two approaches:

- Preferred: calculate our own Elo ratings before every match using the historical match table.
- Reference: compare our values against World Football Elo Ratings from `eloratings.net`.

For World Cup work, national-team Elo is more relevant than ClubElo, which is club-team focused.

## Player, Squad, And Lineup Data

The first confirmed-lineup adapter is implemented with two reproducible sources:

- StatsBomb Open Data for World Cup 2018, Euro 2020, and World Cup 2022 starting
  XIs and positions.
- The CC0 `dcaribou/transfermarkt-datasets` DuckDB export for historical player
  valuations.

The builder matches StatsBomb's player ID, legal name, and official nickname to
Transfermarkt records. It restricts fuzzy candidates by national team and
positional unit, records every match method and score in an audit CSV, and uses
only the latest valuation dated on or before kickoff.

The model-ready features include:

- Changes from the previous starting XI within the tournament.
- Goalkeeper replacement.
- Defensive continuity for the goalkeeper and defenders.
- Attacking continuity for midfielders and attackers.
- Log market-value strength for goalkeeper, defense, midfield, and attack.
- Position-strength changes from the team's previous tournament lineup.
- Identity and valuation coverage indicators.

The current source slice has 179 complete match lineups, a 95.8% identity match
rate, and 90.0% dated-valuation coverage. This is enough to implement and audit
the feature path, but only 115 confirmed-lineup matches precede World Cup 2022.
The V4 correction therefore remains experimental after failing to improve the
2022 holdout.

EA/FIFA-style ratings remain a possible second provider. They should be imported
through the same dated player-strength interface and compared with market value,
not scraped silently or treated as absolute truth.

## Odds Data

Odds are needed for betting recommendations and ROI backtests. They are not required for the first pure prediction model.

Candidate sources:

- Football-Data.co.uk for free historical league odds and odds-field formats.
- The Odds API for paid historical odds snapshots from June 2020 onward.
- API-Football or Sportmonks for broader fixtures, odds, and World Cup coverage.

Free, complete historical World Cup market odds are harder to find than match results. The first version should support odds imports from CSV so we can plug in whichever clean source we choose.

The free odds phase now starts with Football-Data.co.uk club CSVs. This is not the final World Cup model; it is the laboratory for the betting machinery:

- Download ten comparable club leagues with historical opening and closing 1X2 and Over/Under 2.5 odds.
- Train only on older seasons.
- Hold out a later season.
- Compare model probabilities to no-vig bookmaker probabilities.
- Place simulated flat-stake bets only when `model_probability * decimal_odds - 1` clears an edge threshold.
- Add a bookmaker-blend calibration gate so the system can decide to make no bets when the standalone model is weaker than the market.

This lets us test whether the project can recognize value bets before buying national-team odds. Paid international odds should be downloaded later only for the exact tournaments, markets, regions/bookmakers, and pre-game snapshots we plan to evaluate.

The first free feature set uses fields already present in the Football-Data CSVs:

- Opening 1X2 and Over/Under 2.5 odds.
- Final result and total goals.
- Pre-match team Elo.
- Rolling goals for/against.
- Rolling shots for/against.
- Rolling shots on target for/against.
- Rest days since each team's previous match.
- Recent overall form, home-team home form, and away-team away form.

The next external data adapters should add:

- xG for/against from a source that clearly allows historical use.
- Squad value or player ratings.
- Confirmed lineups.
- Injuries and suspensions.

Until those data sources are selected, the model should not fabricate those features. Missing xG columns are represented as neutral zero-value placeholders so future imports can plug in without changing the model interface.

The residual value-model experiment is implemented in `scripts/backtest_residual_value.py`. It builds walk-forward candidate bets so the second-stage model trains on base-model predictions from seasons the base model had not seen. This avoids training the value filter on in-sample base predictions.

The current CLV check uses Football-Data's opening and closing odds columns. A positive decimal CLV means the selected opening price was better than the closing price for the same outcome. The project now supports constrained selection with:

- Maximum odds, currently `4.00`.
- Maximum bets per club weekend, currently `3`.
- Weekend groups stored as Tuesday-Monday periods so Friday-Monday club fixtures stay together.
- Value-mode ranking by selected edge.
- Conservative-mode ranking by selected probability, with a 55% probability floor and 2.50 odds cap.

The expanded free validation dataset covers 2019-2020 through 2025-2026 for the Premier League, Championship, La Liga, Serie A, Bundesliga, Ligue 1, Eredivisie, Primeira Liga, Belgian Pro League, and Scottish Premiership. Every rebuild writes a league-season coverage CSV, and `scripts/audit_club_data_coverage.py` creates the tracked audit report.

## Conservative V2 Calibration

The bookmaker-anchored conservative experiment is implemented in `scripts/backtest_conservative_v2.py`. It keeps the V1 residual model intact as a benchmark and adds a stricter probability and selection layer:

- Calibrated probability is a weighted average of residual-model probability and bookmaker no-vig opening probability.
- Residual weights are selected separately for `1X2` and `Over/Under 2.5` on validation log loss, with a maximum residual weight of 50%.
- Edge thresholds are selected separately by market on the immediately preceding validation season.
- A market stays inactive unless at least 20 validation bets are available and their validation profit is positive.
- Selections require at least 55% calibrated probability and odds no higher than 2.50.
- The policy permits one bet per market per weekend, one bet per match, and at most three bets overall per weekend.

Across the 2022-2023 through 2025-2026 rolling test seasons, V2 placed 33 bets for +9.30 units. All bets came from the first holdout; the following three seasons abstained. This is useful evidence that anchoring can prevent V1's later losses, but it is not enough activity or temporal coverage to claim a deployable edge. The frozen results are in `reports/conservative_v2_validation_summary.md`.

## Train, Validation, Test Split

Use chronological splits, not random splits:

- Train: older matches
- Validation: later matches used for model choices
- Test: newest matches, untouched until final evaluation

Example first split:

- Train: matches through `2018-12-31`
- Validation: `2019-01-01` through `2022-12-31`
- Test: `2023-01-01` onward

For World Cup-specific evaluation, we can also do tournament holdouts:

- Train before World Cup 2018, test on World Cup 2018.
- Train before World Cup 2022, test on World Cup 2022.

The first tournament holdout implementation is strict: test rows use the pre-tournament Elo state and do not update Elo with earlier matches inside the same World Cup. This protects the holdout from label leakage. A later "live tournament" mode can intentionally update team ratings after each completed tournament match.

This protects the project from data leakage and gives a more honest estimate of future performance.
