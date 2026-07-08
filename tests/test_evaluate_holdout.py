import unittest

import pandas as pd

from scripts.evaluate_holdout import apply_target_filter


class EvaluateHoldoutTests(unittest.TestCase):
    def test_clean_3way_uses_market_scores(self):
        frame = pd.DataFrame(
            [
                {
                    "home_goals": 2,
                    "away_goals": 1,
                    "result": "1",
                    "market_3way_home_goals": 1,
                    "market_3way_away_goals": 1,
                    "market_3way_result": "X",
                    "is_clean_3way_result": True,
                }
            ]
        )

        train, test = apply_target_filter(frame, frame, "clean_3way")

        self.assertEqual(train.loc[0, "home_goals"], 1)
        self.assertEqual(train.loc[0, "away_goals"], 1)
        self.assertEqual(train.loc[0, "result"], "X")
        self.assertEqual(test.loc[0, "result"], "X")

    def test_qualification_normalizes_float_like_labels(self):
        frame = pd.DataFrame(
            [
                {
                    "is_world_cup_knockout_stage": True,
                    "qualification_result": "2.0",
                    "result": "X",
                }
            ]
        )

        train, test = apply_target_filter(frame, frame, "qualification")

        self.assertEqual(train.loc[0, "result"], "2")
        self.assertEqual(test.loc[0, "result"], "2")


if __name__ == "__main__":
    unittest.main()
