import unittest

import pandas as pd

from soccer_predictor.club_features import add_club_pre_match_features


class ClubFeatureTests(unittest.TestCase):
    def test_pre_match_features_do_not_use_current_match(self):
        matches = pd.DataFrame(
            [
                _row(1, "2024-01-01", "A", "B", 2, 0, 12, 6, 5, 2),
                _row(2, "2024-01-08", "A", "C", 1, 1, 10, 8, 4, 3),
            ]
        )

        features = add_club_pre_match_features(matches)

        self.assertEqual(features.loc[0, "home_points_per_match_last5_pre"], 1.0)
        self.assertEqual(features.loc[1, "home_points_per_match_last5_pre"], 3.0)
        self.assertEqual(features.loc[1, "home_shots_for_last5_pre"], 12.0)
        self.assertEqual(features.loc[1, "home_sot_for_last5_pre"], 5.0)
        self.assertEqual(features.loc[1, "home_rest_days_pre"], 7.0)


def _row(match_id, date, home, away, home_goals, away_goals, home_shots, away_shots, home_sot, away_sot):
    return {
        "match_id": match_id,
        "date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_shots_on_target": home_sot,
        "away_shots_on_target": away_sot,
        "result": "1" if home_goals > away_goals else "X",
        "neutral": False,
    }


if __name__ == "__main__":
    unittest.main()
