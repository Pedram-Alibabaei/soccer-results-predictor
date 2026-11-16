import unittest

import numpy as np
import pandas as pd

from soccer_predictor.contextual_goals import (
    ContextualGoalModel,
    StageAdjustment,
    fit_negative_binomial_dispersion,
    fit_third_place_adjustment,
)


class StubGoalModel:
    def predict_expected_goals(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "home_xg": np.full(len(frame), 1.5),
                "away_xg": np.full(len(frame), 1.0),
            },
            index=frame.index,
        )


class ContextualGoalTests(unittest.TestCase):
    def test_third_place_multiplier_only_changes_context_match(self):
        model = ContextualGoalModel(
            base_model=StubGoalModel(),
            third_place_adjustment=StageAdjustment(1.2, 4, 14.0, 11.0, 30.0),
            dispersion=0.1,
        )
        frame = pd.DataFrame(
            {
                "home_team": ["A", "C"],
                "away_team": ["B", "D"],
                "is_world_cup_third_place": [False, True],
            }
        )

        expected = model.predict_expected_goals(frame)

        self.assertAlmostEqual(expected.loc[0, "home_xg"], 1.5)
        self.assertAlmostEqual(expected.loc[1, "home_xg"], 1.8)
        self.assertAlmostEqual(expected.loc[1, "away_xg"], 1.2)

    def test_stage_adjustment_excludes_history_after_training_cutoff(self):
        train = pd.DataFrame(
            {
                "date": pd.to_datetime(["2010-07-10", "2014-07-12"]),
                "home_team": ["Uruguay", "Brazil"],
                "away_team": ["Germany", "Netherlands"],
            }
        )
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2010-07-10", "2014-07-12", "2018-07-14"]),
                "home_team": ["Uruguay", "Brazil", "Belgium"],
                "away_team": ["Germany", "Netherlands", "England"],
                "regulation_home_goals": [2, 0, 2],
                "regulation_away_goals": [3, 3, 0],
            }
        )

        adjustment = fit_third_place_adjustment(
            train,
            history,
            StubGoalModel(),
            prior_exposure=10.0,
            half_life_days=None,
        )

        self.assertEqual(adjustment.match_count, 2)
        self.assertAlmostEqual(adjustment.observed_goals, 8.0)
        self.assertAlmostEqual(adjustment.predicted_goals, 5.0)
        self.assertAlmostEqual(adjustment.multiplier, 1.2)

    def test_dispersion_fit_prefers_heavier_tail_for_extreme_counts(self):
        actual = np.array([0, 0, 1, 5, 6, 7], dtype=float)
        expected = np.full(len(actual), 3.0)

        dispersion = fit_negative_binomial_dispersion(
            actual,
            expected,
            grid=(0.0, 0.1, 0.3, 0.6),
        )

        self.assertGreater(dispersion, 0.0)

    def test_risk_output_contains_extreme_total_probabilities(self):
        model = ContextualGoalModel(
            base_model=StubGoalModel(),
            third_place_adjustment=StageAdjustment(1.2, 4, 14.0, 11.0, 30.0),
            dispersion=0.3,
        )
        frame = pd.DataFrame(
            {
                "home_team": ["A"],
                "away_team": ["B"],
                "is_world_cup_third_place": [True],
            }
        )

        risk = model.predict_risk(frame).iloc[0]

        self.assertGreater(risk["probability_6_plus"], 0.0)
        self.assertGreater(risk["probability_10_plus"], 0.0)
        self.assertGreater(risk["dispersion_index"], 1.0)


if __name__ == "__main__":
    unittest.main()
