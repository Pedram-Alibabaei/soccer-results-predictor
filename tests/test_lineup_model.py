import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from soccer_predictor.lineup_model import (
    LINEUP_ADJUSTMENT_FEATURE_COLUMNS,
    fit_lineup_adjusted_goal_model,
)
from soccer_predictor.modeling import fit_poisson_goal_model


class LineupAdjustedModelTests(unittest.TestCase):
    def test_zero_strength_matches_baseline_and_missing_lineup_bypasses_adjustment(self):
        train = pd.DataFrame([_row(index) for index in range(36)])
        baseline = fit_poisson_goal_model(
            train,
            epochs=30,
            learning_rate=0.02,
            half_life_days=None,
        )
        adjusted = fit_lineup_adjusted_goal_model(
            baseline,
            train,
            epochs=30,
            learning_rate=0.02,
            l2=1.0,
            half_life_days=None,
        )
        zero_strength = replace(adjusted, correction_strength=0.0)

        baseline_xg = baseline.predict_expected_goals(train)
        zero_xg = zero_strength.predict_expected_goals(train)
        self.assertTrue(np.allclose(baseline_xg, zero_xg))

        unavailable = train.iloc[[0]].copy()
        unavailable["has_confirmed_lineups"] = 0.0
        unavailable_xg = adjusted.predict_expected_goals(unavailable)
        unavailable_baseline_xg = baseline.predict_expected_goals(unavailable)
        self.assertTrue(np.allclose(unavailable_xg, unavailable_baseline_xg))

    def test_adjusted_probabilities_sum_to_one(self):
        train = pd.DataFrame([_row(index) for index in range(36)])
        baseline = fit_poisson_goal_model(train, epochs=20, half_life_days=None)
        adjusted = fit_lineup_adjusted_goal_model(
            baseline,
            train,
            epochs=20,
            l2=1.0,
            half_life_days=None,
        )

        probabilities = adjusted.predict_proba(train.iloc[:3])

        self.assertTrue((probabilities.sum(axis=1).round(10) == 1.0).all())


def _row(index):
    result_cycle = [(2, 0, "1"), (1, 1, "X"), (0, 2, "2")]
    home_goals, away_goals, result = result_cycle[index % len(result_cycle)]
    row = {
        "date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=index),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_elo_pre": 1600 + (index % 3) * 20,
        "away_elo_pre": 1580,
        "elo_diff_pre": 20 + (index % 3) * 20,
        "home_attack_pre": 1.5,
        "away_attack_pre": 1.2,
        "home_defense_pre": 1.0,
        "away_defense_pre": 1.1,
        "home_matches_played_pre": 50,
        "away_matches_played_pre": 50,
        "neutral": True,
        "result": result,
        "has_confirmed_lineups": 1.0,
    }
    for column in LINEUP_ADJUSTMENT_FEATURE_COLUMNS:
        row[column] = float((index + len(column)) % 5) / 4.0
    return row


if __name__ == "__main__":
    unittest.main()
