import unittest

import numpy as np
import pandas as pd

from soccer_predictor.modeling import (
    evaluate_predictions,
    fit_dixon_coles_goal_model,
    fit_draw_aware_poisson_model,
    fit_poisson_goal_model,
    fit_softmax_baseline,
)


class ModelingTests(unittest.TestCase):
    def test_softmax_baseline_outputs_probabilities_that_sum_to_one(self):
        train = pd.DataFrame(
            [
                _row("2020-01-01", 1700, 1500, "1"),
                _row("2020-01-02", 1500, 1500, "X"),
                _row("2020-01-03", 1450, 1650, "2"),
                _row("2020-01-04", 1800, 1500, "1"),
                _row("2020-01-05", 1500, 1700, "2"),
                _row("2020-01-06", 1500, 1500, "X"),
            ]
        )
        test = train.copy()

        model = fit_softmax_baseline(train, epochs=25, half_life_days=None)
        probabilities = model.predict_proba(test)

        self.assertEqual(probabilities.shape, (6, 3))
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all().all())
        self.assertTrue((probabilities.sum(axis=1).round(10) == 1.0).all())

    def test_evaluate_predictions_returns_metrics(self):
        actual = pd.Series(["1", "X", "2"])
        probabilities = pd.DataFrame(
            {
                "1": [0.7, 0.2, 0.2],
                "X": [0.2, 0.6, 0.2],
                "2": [0.1, 0.2, 0.6],
            }
        )

        metrics = evaluate_predictions(actual, probabilities)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertLess(metrics["log_loss"], 1.0)

    def test_poisson_goal_model_outputs_probabilities_and_expected_goals(self):
        train = pd.DataFrame(
            [
                _row("2020-01-01", 1700, 1500, "1", home_goals=2, away_goals=0),
                _row("2020-01-02", 1500, 1500, "X", home_goals=1, away_goals=1),
                _row("2020-01-03", 1450, 1650, "2", home_goals=0, away_goals=2),
                _row("2020-01-04", 1800, 1500, "1", home_goals=3, away_goals=1),
                _row("2020-01-05", 1500, 1700, "2", home_goals=1, away_goals=3),
                _row("2020-01-06", 1500, 1500, "X", home_goals=0, away_goals=0),
            ]
        )

        model = fit_poisson_goal_model(train, epochs=25, learning_rate=0.02, half_life_days=None)
        expected_goals = model.predict_expected_goals(train)
        probabilities = model.predict_proba(train)

        self.assertEqual(expected_goals.shape, (6, 2))
        self.assertTrue((expected_goals > 0).all().all())
        self.assertEqual(probabilities.shape, (6, 3))
        self.assertTrue((probabilities.sum(axis=1).round(10) == 1.0).all())

    def test_world_cup_context_features_default_to_zero_when_absent(self):
        train = pd.DataFrame(
            [
                _row("2020-01-01", 1700, 1500, "1"),
                _row("2020-01-02", 1500, 1500, "X"),
                _row("2020-01-03", 1450, 1650, "2"),
            ]
        )

        model = fit_softmax_baseline(train, epochs=10, half_life_days=None)
        probabilities = model.predict_proba(train)

        self.assertEqual(probabilities.shape, (3, 3))

    def test_dixon_coles_goal_model_fits_rho_from_training_data(self):
        train = pd.DataFrame(
            [
                _row("2020-01-01", 1700, 1500, "1", home_goals=1, away_goals=0),
                _row("2020-01-02", 1500, 1500, "X", home_goals=0, away_goals=0),
                _row("2020-01-03", 1450, 1650, "2", home_goals=0, away_goals=1),
                _row("2020-01-04", 1800, 1500, "X", home_goals=1, away_goals=1),
                _row("2020-01-05", 1500, 1700, "X", home_goals=0, away_goals=0),
                _row("2020-01-06", 1500, 1500, "X", home_goals=1, away_goals=1),
            ]
        )

        model = fit_dixon_coles_goal_model(
            train,
            epochs=25,
            learning_rate=0.02,
            half_life_days=None,
        )

        self.assertTrue(np.isfinite(model.dixon_coles_rho))

    def test_draw_aware_poisson_model_outputs_probabilities(self):
        train = pd.DataFrame(
            [
                _row("2020-01-01", 1700, 1500, "1", home_goals=2, away_goals=0),
                _row("2020-01-02", 1500, 1500, "X", home_goals=0, away_goals=0),
                _row("2020-01-03", 1450, 1650, "2", home_goals=0, away_goals=2),
                _row("2020-01-04", 1800, 1500, "1", home_goals=3, away_goals=1),
                _row("2020-01-05", 1500, 1700, "2", home_goals=1, away_goals=3),
                _row("2020-01-06", 1500, 1500, "X", home_goals=1, away_goals=1),
            ]
        )

        model = fit_draw_aware_poisson_model(
            train,
            epochs=25,
            learning_rate=0.02,
            half_life_days=None,
        )
        probabilities = model.predict_proba(train)

        self.assertEqual(probabilities.shape, (6, 3))
        self.assertTrue((probabilities.sum(axis=1).round(10) == 1.0).all())


def _row(date, home_elo, away_elo, result, home_goals=1, away_goals=0):
    return {
        "date": pd.Timestamp(date),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "elo_diff_pre": home_elo - away_elo,
        "home_attack_pre": 1.5,
        "away_attack_pre": 1.2,
        "home_defense_pre": 1.0,
        "away_defense_pre": 1.1,
        "home_matches_played_pre": 50,
        "away_matches_played_pre": 50,
        "neutral": True,
        "result": result,
    }


if __name__ == "__main__":
    unittest.main()
