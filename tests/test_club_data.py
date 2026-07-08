import unittest

import pandas as pd

from soccer_predictor.club_data import LeagueConfig, process_football_data_frame, season_code


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


if __name__ == "__main__":
    unittest.main()
