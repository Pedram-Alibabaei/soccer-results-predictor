import unittest

import pandas as pd

from soccer_predictor.elo import add_elo_features, expected_result, fit_elo_state


class EloTests(unittest.TestCase):
    def test_first_match_uses_initial_ratings(self):
        matches = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "date": pd.Timestamp("2020-01-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 1,
                    "away_goals": 0,
                    "result": "1",
                    "total_goals": 1,
                    "goal_difference": 1,
                    "both_teams_scored": False,
                    "tournament": "Friendly",
                    "city": "City",
                    "venue_country": "Country",
                    "neutral": True,
                }
            ]
        )

        features = add_elo_features(matches)

        self.assertEqual(features.loc[0, "home_elo_pre"], 1500.0)
        self.assertEqual(features.loc[0, "away_elo_pre"], 1500.0)
        self.assertEqual(features.loc[0, "home_matches_played_pre"], 0)
        self.assertEqual(features.loc[0, "away_matches_played_pre"], 0)

    def test_later_match_sees_previous_result(self):
        matches = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "date": pd.Timestamp("2020-01-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 1,
                    "away_goals": 0,
                    "result": "1",
                    "total_goals": 1,
                    "goal_difference": 1,
                    "both_teams_scored": False,
                    "tournament": "Friendly",
                    "city": "City",
                    "venue_country": "Country",
                    "neutral": True,
                },
                {
                    "match_id": 2,
                    "date": pd.Timestamp("2020-02-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 0,
                    "away_goals": 0,
                    "result": "X",
                    "total_goals": 0,
                    "goal_difference": 0,
                    "both_teams_scored": False,
                    "tournament": "Friendly",
                    "city": "City",
                    "venue_country": "Country",
                    "neutral": True,
                },
            ]
        )

        features = add_elo_features(matches)

        self.assertGreater(features.loc[1, "home_elo_pre"], 1500.0)
        self.assertLess(features.loc[1, "away_elo_pre"], 1500.0)
        self.assertEqual(features.loc[1, "home_matches_played_pre"], 1)
        self.assertEqual(features.loc[1, "away_matches_played_pre"], 1)

    def test_expected_result_is_symmetric(self):
        stronger = expected_result(1600, 1500)
        weaker = expected_result(1500, 1600)

        self.assertAlmostEqual(stronger + weaker, 1.0)
        self.assertGreater(stronger, weaker)

    def test_features_can_be_created_without_updating_initial_state(self):
        train = pd.DataFrame(
            [
                _match(1, "2020-01-01", "A", "B", 2, 0),
            ]
        )
        test = pd.DataFrame(
            [
                _match(2, "2020-02-01", "A", "B", 5, 0),
                _match(3, "2020-03-01", "A", "B", 0, 5),
            ]
        )
        train["date"] = pd.to_datetime(train["date"])
        test["date"] = pd.to_datetime(test["date"])

        state = fit_elo_state(train)
        features = add_elo_features(test, initial_state=state, update_ratings=False)

        self.assertAlmostEqual(features.loc[0, "home_elo_pre"], features.loc[1, "home_elo_pre"])
        self.assertAlmostEqual(features.loc[0, "away_elo_pre"], features.loc[1, "away_elo_pre"])


if __name__ == "__main__":
    unittest.main()


def _match(match_id, date, home_team, away_team, home_goals, away_goals):
    return {
        "match_id": match_id,
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": "1" if home_goals > away_goals else "2" if away_goals > home_goals else "X",
        "total_goals": home_goals + away_goals,
        "goal_difference": home_goals - away_goals,
        "both_teams_scored": home_goals > 0 and away_goals > 0,
        "tournament": "Friendly",
        "city": "City",
        "venue_country": "Country",
        "neutral": True,
    }
