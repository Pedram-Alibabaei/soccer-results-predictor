import unittest

import pandas as pd

from soccer_predictor.data_pipeline import (
    DatasetSplitConfig,
    add_world_cup_context,
    add_qualification_results,
    apply_world_cup_settlements,
    build_world_cup_holdouts,
    process_international_results,
    split_dataset,
)


class DataPipelineTests(unittest.TestCase):
    def test_process_international_results_drops_unplayed_matches(self):
        raw = pd.DataFrame(
            [
                {
                    "date": "2026-07-01",
                    "home_team": "United States",
                    "away_team": "Bosnia and Herzegovina",
                    "home_score": 2,
                    "away_score": 0,
                    "tournament": "FIFA World Cup",
                    "city": "Seattle",
                    "country": "United States",
                    "neutral": False,
                },
                {
                    "date": "2026-07-02",
                    "home_team": "Portugal",
                    "away_team": "Croatia",
                    "home_score": None,
                    "away_score": None,
                    "tournament": "FIFA World Cup",
                    "city": "Dallas",
                    "country": "United States",
                    "neutral": True,
                },
            ]
        )

        processed = process_international_results(raw)

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed.loc[0, "result"], "1")
        self.assertEqual(processed.loc[0, "total_goals"], 2)

    def test_split_dataset_uses_chronological_boundaries(self):
        dataset = pd.DataFrame(
            {
                "match_id": [1, 2, 3],
                "date": pd.to_datetime(["2018-01-01", "2020-01-01", "2024-01-01"]),
            }
        )

        splits = split_dataset(
            dataset,
            DatasetSplitConfig(validation_start="2019-01-01", test_start="2023-01-01"),
        )

        self.assertEqual(splits["train"]["match_id"].tolist(), [1])
        self.assertEqual(splits["validation"]["match_id"].tolist(), [2])
        self.assertEqual(splits["test"]["match_id"].tolist(), [3])

    def test_world_cup_holdout_uses_pre_tournament_test_features(self):
        matches = pd.DataFrame(
            [
                _match(1, "2018-01-01", "A", "B", 2, 0, "Friendly"),
                _match(2, "2018-06-14", "A", "B", 3, 0, "FIFA World Cup"),
                _match(3, "2018-06-20", "A", "B", 0, 3, "FIFA World Cup"),
            ]
        )
        matches["date"] = pd.to_datetime(matches["date"])

        holdouts = build_world_cup_holdouts(matches, years=(2018,))
        holdout = holdouts["world_cup_2018"]

        self.assertEqual(len(holdout["train"]), 1)
        self.assertEqual(len(holdout["test"]), 2)
        self.assertEqual(holdout["test"]["match_id"].tolist(), [2, 3])
        self.assertAlmostEqual(
            holdout["test"].loc[0, "home_elo_pre"],
            holdout["test"].loc[1, "home_elo_pre"],
        )
        self.assertEqual(
            holdout["test"].loc[0, "home_matches_played_pre"],
            holdout["test"].loc[1, "home_matches_played_pre"],
        )

    def test_world_cup_context_marks_knockout_as_not_clean_3way(self):
        rows = []
        for match_number in range(1, 65):
            rows.append(
                _match(
                    match_number,
                    f"2018-06-{1 + ((match_number - 1) // 3):02d}",
                    f"Home {match_number}",
                    f"Away {match_number}",
                    1,
                    0,
                    "FIFA World Cup",
                )
            )
        matches = pd.DataFrame(rows)
        matches["date"] = pd.to_datetime(matches["date"])

        with_context = add_world_cup_context(matches)

        self.assertEqual(with_context.loc[47, "world_cup_stage"], "group")
        self.assertTrue(with_context.loc[47, "is_clean_3way_result"])
        self.assertEqual(with_context.loc[48, "world_cup_stage"], "round_of_16")
        self.assertFalse(with_context.loc[48, "is_clean_3way_result"])
        self.assertEqual(with_context.loc[63, "world_cup_stage"], "final")

    def test_world_cup_settlement_overrides_market_result(self):
        matches = pd.DataFrame(
            [
                _match(1, "2018-07-11", "Croatia", "England", 2, 1, "FIFA World Cup"),
            ]
        )
        matches["date"] = pd.to_datetime(matches["date"])
        matches = add_world_cup_context(matches)
        matches["market_3way_home_goals"] = matches["home_goals"]
        matches["market_3way_away_goals"] = matches["away_goals"]
        matches["market_3way_result"] = matches["result"]
        matches["has_90_minute_settlement"] = False
        matches["qualification_result"] = ""
        matches["settlement_notes"] = ""
        settlements = pd.DataFrame(
            [
                {
                    "date": "2018-07-11",
                    "home_team": "Croatia",
                    "away_team": "England",
                    "market_3way_home_goals": 1,
                    "market_3way_away_goals": 1,
                    "qualification_result": "1",
                    "notes": "Croatia advanced after extra time",
                }
            ]
        )

        settled = apply_world_cup_settlements(matches, settlements)

        self.assertEqual(settled.loc[0, "market_3way_result"], "X")
        self.assertTrue(settled.loc[0, "is_clean_3way_result"])
        self.assertEqual(settled.loc[0, "qualification_result"], "1")

    def test_qualification_results_use_shootout_winner_for_tied_knockout(self):
        matches = pd.DataFrame(
            [
                _match(1, "2018-07-03", "Colombia", "England", 1, 1, "FIFA World Cup"),
            ]
        )
        matches["date"] = pd.to_datetime(matches["date"])
        matches = add_world_cup_context(matches)
        matches["qualification_result"] = ""
        shootouts = pd.DataFrame(
            [
                {
                    "date": "2018-07-03",
                    "home_team": "Colombia",
                    "away_team": "England",
                    "winner": "England",
                }
            ]
        )

        with_qualification = add_qualification_results(matches, shootouts)

        self.assertEqual(with_qualification.loc[0, "qualification_result"], "2")


if __name__ == "__main__":
    unittest.main()


def _match(
    match_id,
    date,
    home_team,
    away_team,
    home_goals,
    away_goals,
    tournament,
):
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
        "tournament": tournament,
        "city": "City",
        "venue_country": "Country",
        "neutral": True,
    }
