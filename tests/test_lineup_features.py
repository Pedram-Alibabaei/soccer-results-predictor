import unittest

import pandas as pd

from soccer_predictor.lineup_features import (
    CONFIRMED_LINEUP_FEATURE_COLUMNS,
    attach_lineup_features,
    build_match_lineup_features,
    match_statsbomb_players,
    normalize_name,
    parse_statsbomb_starters,
    position_group,
)


class LineupFeatureTests(unittest.TestCase):
    def test_name_and_position_normalization(self):
        self.assertEqual(normalize_name("Mark-Anthony Kaye"), "mark anthony kaye")
        self.assertEqual(normalize_name("Steven Vitória"), "steven vitoria")
        self.assertEqual(position_group("Right Center Back"), "defense")
        self.assertEqual(position_group("Left Defensive Midfield"), "midfield")
        self.assertEqual(position_group("Right Wing"), "attack")

    def test_parse_statsbomb_starters_ignores_substitutes(self):
        match = {
            "match_id": 10,
            "match_date": "2022-12-01",
            "home_team": {"home_team_name": "Morocco"},
            "away_team": {"away_team_name": "Canada"},
            "home_score": 2,
            "away_score": 1,
            "competition_stage": {"name": "Group Stage"},
        }
        payload = [
            {
                "team_name": "Morocco",
                "lineup": [
                    {
                        "player_id": 1,
                        "player_name": "Starter",
                        "positions": [
                            {
                                "position": "Goalkeeper",
                                "from": "00:00",
                                "start_reason": "Starting XI",
                            }
                        ],
                    },
                    {
                        "player_id": 2,
                        "player_name": "Substitute",
                        "positions": [
                            {
                                "position": "Center Forward",
                                "from": "70:00",
                                "start_reason": "Substitution - On (Tactical)",
                            }
                        ],
                    },
                ],
            }
        ]

        starters = parse_statsbomb_starters(
            match,
            payload,
            competition_name="FIFA World Cup",
            season_name="2022",
        )

        self.assertEqual(len(starters), 1)
        self.assertEqual(starters.iloc[0]["player_name"], "Starter")
        self.assertEqual(starters.iloc[0]["side"], "home")

    def test_rotation_and_unit_continuity_use_previous_tournament_lineup(self):
        rows = []
        rows.extend(_lineup_rows(1, "2022-11-20", "home", "Team A", list(range(1, 12))))
        rows.extend(_lineup_rows(1, "2022-11-20", "away", "Team B", list(range(21, 32))))
        changed_home = [12, 13, 3, 4, 5, 6, 7, 8, 14, 10, 11]
        rows.extend(_lineup_rows(2, "2022-11-24", "home", "Team A", changed_home))
        rows.extend(_lineup_rows(2, "2022-11-24", "away", "Team B", list(range(21, 32))))

        features = build_match_lineup_features(pd.DataFrame(rows))
        first = features.iloc[0]
        second = features.iloc[1]

        self.assertEqual(first["home_previous_lineup_available_pre"], 0.0)
        self.assertEqual(first["home_defensive_continuity_pre"], 1.0)
        self.assertEqual(second["home_lineup_changes_pre"], 3.0)
        self.assertEqual(second["home_goalkeeper_changed_pre"], 1.0)
        self.assertAlmostEqual(second["home_defensive_continuity_pre"], 3 / 5)
        self.assertAlmostEqual(second["home_attacking_continuity_pre"], 5 / 6)
        self.assertEqual(second["away_lineup_changes_pre"], 0.0)

    def test_player_matching_accepts_exact_and_token_subset_names(self):
        starters = pd.DataFrame(
            [
                {
                    "statsbomb_player_id": 1,
                    "player_name": "Achraf Hakimi Mouh",
                    "team": "Morocco",
                },
                {
                    "statsbomb_player_id": 2,
                    "player_name": "Mark Anthony Kaye",
                    "team": "Canada",
                },
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "player_id": 100,
                    "name": "Achraf Hakimi",
                    "first_name": "Achraf",
                    "last_name": "Hakimi",
                    "position": "Defender",
                    "sub_position": "Right-Back",
                    "country_of_citizenship": "Morocco",
                },
                {
                    "player_id": 200,
                    "name": "Mark-Anthony Kaye",
                    "first_name": "Mark-Anthony",
                    "last_name": "Kaye",
                    "position": "Midfield",
                    "sub_position": "Central Midfield",
                    "country_of_citizenship": "Canada",
                },
            ]
        )

        matches = match_statsbomb_players(starters, players)

        self.assertEqual(matches["transfermarkt_player_id"].tolist(), [100, 200])
        self.assertEqual(matches["player_match_method"].tolist(), ["token_subset", "exact"])

    def test_player_matching_prefers_statsbomb_nickname(self):
        starters = pd.DataFrame(
            [
                {
                    "statsbomb_player_id": 1,
                    "player_name": "Carlos Henrique Casimiro",
                    "player_nickname": "Casemiro",
                    "team": "Brazil",
                }
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "player_id": 100,
                    "name": "Casemiro",
                    "first_name": "Carlos Henrique",
                    "last_name": "Casimiro",
                    "position": "Midfield",
                    "sub_position": "Defensive Midfield",
                    "country_of_citizenship": "Brazil",
                }
            ]
        )

        matches = match_statsbomb_players(starters, players)

        self.assertEqual(matches.iloc[0]["transfermarkt_player_id"], 100)
        self.assertEqual(matches.iloc[0]["player_match_method"], "nickname_exact")

    def test_attach_lineup_features_marks_uncovered_matches(self):
        matches = pd.DataFrame(
            [
                {"date": "2022-12-01", "home_team": "Morocco", "away_team": "Canada"},
                {"date": "2022-12-02", "home_team": "Brazil", "away_team": "Cameroon"},
            ]
        )
        covered = {
            "date": "2022-12-01",
            "home_team": "Morocco",
            "away_team": "Canada",
        }
        for column in CONFIRMED_LINEUP_FEATURE_COLUMNS:
            covered[column] = 1.0

        result = attach_lineup_features(matches, pd.DataFrame([covered]))

        self.assertEqual(result.iloc[0]["has_confirmed_lineups"], 1.0)
        self.assertEqual(result.iloc[1]["has_confirmed_lineups"], 0.0)
        self.assertEqual(result.iloc[1]["home_defensive_continuity_pre"], 1.0)

    def test_attach_lineup_features_swaps_neutral_match_orientation(self):
        matches = pd.DataFrame(
            [{"date": "2022-12-01", "home_team": "Canada", "away_team": "Morocco"}]
        )
        covered = {
            "date": "2022-12-01",
            "home_team": "Morocco",
            "away_team": "Canada",
        }
        for column in CONFIRMED_LINEUP_FEATURE_COLUMNS:
            covered[column] = 0.0
        covered.update(
            {
                "has_confirmed_lineups": 1.0,
                "home_lineup_changes_pre": 3.0,
                "away_lineup_changes_pre": 1.0,
                "lineup_changes_diff_pre": 2.0,
            }
        )

        result = attach_lineup_features(matches, pd.DataFrame([covered]))

        self.assertEqual(result.iloc[0]["has_confirmed_lineups"], 1.0)
        self.assertEqual(result.iloc[0]["home_lineup_changes_pre"], 1.0)
        self.assertEqual(result.iloc[0]["away_lineup_changes_pre"], 3.0)
        self.assertEqual(result.iloc[0]["lineup_changes_diff_pre"], -2.0)


def _lineup_rows(match_id, date, side, team, player_ids):
    groups = ["goalkeeper"] + ["defense"] * 4 + ["midfield"] * 3 + ["attack"] * 3
    opponent = "Team B" if team == "Team A" else "Team A"
    return [
        {
            "statsbomb_match_id": match_id,
            "date": date,
            "competition_name": "Test Cup",
            "season_name": "2022",
            "home_team": team if side == "home" else opponent,
            "away_team": opponent if side == "home" else team,
            "team": team,
            "side": side,
            "statsbomb_player_id": player_id,
            "position_group": group,
            "market_value_in_eur": 10_000_000 + player_id,
        }
        for player_id, group in zip(player_ids, groups, strict=True)
    ]


if __name__ == "__main__":
    unittest.main()
