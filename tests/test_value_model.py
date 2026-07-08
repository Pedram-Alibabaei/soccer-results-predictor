import unittest

import pandas as pd

from soccer_predictor.value_model import (
    build_bet_candidates,
    fit_clv_model,
    fit_residual_value_model,
    select_value_bets,
    summarize_combined_thresholds,
)


class ValueModelTests(unittest.TestCase):
    def test_build_bet_candidates_creates_one_row_per_outcome(self):
        matches = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "date": pd.Timestamp("2024-01-01"),
                    "season_start_year": 2024,
                    "season": "2024-2025",
                    "league_name": "League",
                    "home_team": "Home",
                    "away_team": "Away",
                    "result": "1",
                    "total_goals": 3,
                    "odds_open_1": 2.0,
                    "odds_open_X": 3.5,
                    "odds_open_2": 4.0,
                    "odds_open_over_2_5": 1.9,
                    "odds_open_under_2_5": 1.9,
                }
            ]
        )
        probabilities = {
            "1x2": pd.DataFrame({"1": [0.55], "X": [0.25], "2": [0.20]}, index=matches.index),
            "over_under_2_5": pd.DataFrame({"+": [0.60], "-": [0.40]}, index=matches.index),
        }
        odds_columns = {
            "1x2": {"1": "odds_open_1", "X": "odds_open_X", "2": "odds_open_2"},
            "over_under_2_5": {
                "+": "odds_open_over_2_5",
                "-": "odds_open_under_2_5",
            },
        }

        candidates = build_bet_candidates(matches, probabilities, odds_columns)

        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates["won"].sum(), 2)

    def test_residual_value_model_scores_and_selects_bets(self):
        candidates = pd.DataFrame(
            [
                _candidate(1, "1x2", "1", True, 0.65, 0.50, 2.2),
                _candidate(1, "1x2", "X", False, 0.20, 0.28, 3.4),
                _candidate(1, "1x2", "2", False, 0.15, 0.22, 4.5),
                _candidate(2, "1x2", "1", False, 0.30, 0.50, 1.8),
                _candidate(2, "1x2", "X", False, 0.20, 0.25, 3.5),
                _candidate(2, "1x2", "2", True, 0.50, 0.25, 4.2),
            ]
        )

        model = fit_residual_value_model(candidates, epochs=25, learning_rate=0.05)
        scored = model.score_candidates(candidates)
        selected = select_value_bets(scored, min_edge=-1.0)

        sums = scored.groupby(["match_id", "market"])["value_probability"].sum().round(10)
        self.assertTrue((sums == 1.0).all())
        self.assertGreater(len(selected), 0)

    def test_clv_model_scores_candidates_and_combined_thresholds(self):
        candidates = pd.DataFrame(
            [
                _candidate(1, "1x2", "1", True, 0.65, 0.50, 2.2, clv_decimal=0.04),
                _candidate(1, "1x2", "X", False, 0.20, 0.28, 3.4, clv_decimal=-0.02),
                _candidate(1, "1x2", "2", False, 0.15, 0.22, 4.5, clv_decimal=-0.03),
                _candidate(2, "1x2", "1", False, 0.30, 0.50, 1.8, clv_decimal=-0.01),
                _candidate(2, "1x2", "X", False, 0.20, 0.25, 3.5, clv_decimal=0.02),
                _candidate(2, "1x2", "2", True, 0.50, 0.25, 4.2, clv_decimal=0.05),
            ]
        )

        residual_model = fit_residual_value_model(candidates, epochs=25, learning_rate=0.05)
        clv_model = fit_clv_model(candidates, epochs=25, learning_rate=0.05)
        scored = clv_model.score_clv_candidates(residual_model.score_candidates(candidates))
        summary = summarize_combined_thresholds(
            scored,
            [0.0],
            [0.0],
            edge_column="value_edge",
            probability_column="value_probability",
            stake=1.0,
            max_decimal_odds=4.0,
        )
        selected = select_value_bets(
            scored,
            min_edge=-1.0,
            secondary_probability_column="clv_model_probability",
            min_secondary_probability=0.0,
        )

        self.assertIn("clv_model_probability", scored.columns)
        self.assertIn("min_clv_model_probability", summary.columns)
        self.assertGreater(len(selected), 0)


def _candidate(
    match_id,
    market,
    outcome,
    won,
    model_probability,
    bookmaker_probability,
    decimal_odds,
    clv_decimal=0.0,
):
    return {
        "match_id": match_id,
        "date": pd.Timestamp("2024-01-01"),
        "season_start_year": 2024,
        "season": "2024-2025",
        "league": "League",
        "home_team": "Home",
        "away_team": "Away",
        "market": market,
        "outcome": outcome,
        "actual_result": outcome if won else "other",
        "model_probability": model_probability,
        "bookmaker_probability": bookmaker_probability,
        "probability_gap": model_probability - bookmaker_probability,
        "abs_probability_gap": abs(model_probability - bookmaker_probability),
        "decimal_odds": decimal_odds,
        "closing_decimal_odds": decimal_odds / (1.0 + clv_decimal),
        "clv_decimal": clv_decimal,
        "clv_probability": clv_decimal,
        "fair_odds": 1.0 / model_probability,
        "raw_edge": model_probability * decimal_odds - 1.0,
        "won": won,
        "profit": decimal_odds - 1.0 if won else -1.0,
    }


if __name__ == "__main__":
    unittest.main()
