import unittest

import pandas as pd

from soccer_predictor.club_data import (
    LeagueConfig,
    process_football_data_frame,
    season_code,
    summarize_club_data_coverage,
)


class ClubDataTests(unittest.TestCase):
    def test_season_code(self):
        self.assertEqual(season_code(2024), "2425")

    def test_process_football_data_frame_normalizes_results_and_odds(self):
        source = pd.DataFrame(
            [
                {
                    "Date": "16/08/2024",
                    "Time": "20:00",
                    "HomeTeam": "Home FC",
                    "AwayTeam": "Away FC",
                    "FTHG": 2,
                    "FTAG": 1,
                    "FTR": "H",
                    "AvgCH": 1.80,
                    "AvgCD": 3.70,
                    "AvgCA": 4.90,
                    "AvgC>2.5": 1.95,
                    "AvgC<2.5": 1.85,
                }
            ]
        )

        matches = process_football_data_frame(
            source,
            league=LeagueConfig("T1", "Test League", "Testland"),
            season_start_year=2024,
        )

        self.assertEqual(matches.loc[0, "result"], "1")
        self.assertTrue(matches.loc[0, "has_1x2_odds"])
        self.assertTrue(matches.loc[0, "has_ou_2_5_odds"])
        self.assertEqual(matches.loc[0, "odds_1x2_source"], "AvgC")
        self.assertEqual(matches.loc[0, "odds_ou_2_5_source"], "AvgC")

    def test_coverage_summary_reports_opening_closing_and_feature_rates(self):
        source = pd.DataFrame(
            [
                {
                    "Date": "16/08/2024",
                    "HomeTeam": "Home FC",
                    "AwayTeam": "Away FC",
                    "FTHG": 2,
                    "FTAG": 1,
                    "FTR": "H",
                    "HS": 12,
                    "AS": 9,
                    "HST": 5,
                    "AST": 3,
                    "AvgH": 2.0,
                    "AvgD": 3.4,
                    "AvgA": 3.8,
                    "AvgCH": 1.9,
                    "AvgCD": 3.5,
                    "AvgCA": 4.0,
                    "Avg>2.5": 1.9,
                    "Avg<2.5": 1.9,
                    "AvgC>2.5": 1.85,
                    "AvgC<2.5": 1.95,
                }
            ]
        )
        matches = process_football_data_frame(
            source,
            league=LeagueConfig("T1", "Test League", "Testland"),
            season_start_year=2024,
        )

        coverage = summarize_club_data_coverage(matches)

        self.assertEqual(coverage.loc[0, "matches"], 1)
        self.assertEqual(coverage.loc[0, "open_1x2_rate"], 1.0)
        self.assertEqual(coverage.loc[0, "close_ou_2_5_rate"], 1.0)
        self.assertEqual(coverage.loc[0, "shot_stat_rate"], 1.0)
        self.assertEqual(coverage.loc[0, "fully_comparable_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
