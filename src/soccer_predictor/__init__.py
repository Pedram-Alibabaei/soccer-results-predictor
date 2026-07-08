"""Soccer Results Predictor package."""

from .markets import (
    DrawNoBetProbabilities,
    ScoreProbabilityTable,
    both_teams_to_score,
    correct_score,
    double_chance,
    draw_no_bet,
    european_handicap_3way,
    expected_value,
    fair_decimal_odds,
    match_result_and_over_under,
    over_under,
    remove_vig,
    score_matrix,
    three_way,
)

__all__ = [
    "DrawNoBetProbabilities",
    "ScoreProbabilityTable",
    "both_teams_to_score",
    "correct_score",
    "double_chance",
    "draw_no_bet",
    "european_handicap_3way",
    "expected_value",
    "fair_decimal_odds",
    "match_result_and_over_under",
    "over_under",
    "remove_vig",
    "score_matrix",
    "three_way",
]
