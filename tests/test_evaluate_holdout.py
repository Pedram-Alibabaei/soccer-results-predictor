import unittest

import pandas as pd

from scripts.evaluate_holdout import apply_live_tournament_ratings, apply_target_filter


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

    def test_live_tournament_ratings_update_between_test_matches(self):
        train = pd.DataFrame(
            [
                _match(1, "2026-01-01", "A", "B", 1, 0),
            ]
        )
        test = pd.DataFrame(
            [
                _match(2, "2026-06-01", "A", "B", 3, 0),
                _match(3, "2026-06-05", "A", "B", 0, 0),
            ]
        )
        train["date"] = pd.to_datetime(train["date"])
        test["date"] = pd.to_datetime(test["date"])

        live = apply_live_tournament_ratings(train, test)

        self.assertEqual(live.loc[0, "home_matches_played_pre"], 1)
        self.assertEqual(live.loc[1, "home_matches_played_pre"], 2)
        self.assertGreater(live.loc[1, "home_elo_pre"], live.loc[0, "home_elo_pre"])


if __name__ == "__main__":
    unittest.main()


def _match(match_id, date, home_team, away_team, home_goals, away_goals):
    goal_difference = home_goals - away_goals
    if goal_difference > 0:
        result = "1"
    elif goal_difference < 0:
        result = "2"
    else:
        result = "X"

    return {
        "match_id": match_id,
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "total_goals": home_goals + away_goals,
        "goal_difference": goal_difference,
        "both_teams_scored": home_goals > 0 and away_goals > 0,
        "tournament": "FIFA World Cup",
        "city": "City",
        "venue_country": "Country",
        "neutral": True,
    }
