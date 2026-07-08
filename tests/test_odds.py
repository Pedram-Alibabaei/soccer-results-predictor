import unittest

import pandas as pd

from soccer_predictor.odds import (
    bookmaker_probabilities,
    find_value_bets_1x2,
    kelly_fraction,
    summarize_bets,
)


class OddsTests(unittest.TestCase):
    def test_bookmaker_probabilities_remove_vig(self):
        frame = pd.DataFrame([{"odds_1": 1.80, "odds_X": 3.80, "odds_2": 5.00}])

        probabilities = bookmaker_probabilities(frame, {"1": "odds_1", "X": "odds_X", "2": "odds_2"})

        self.assertAlmostEqual(probabilities.loc[0].sum(), 1.0, places=12)

    def test_find_value_bets_settles_profit(self):
        matches = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "home_team": "Home",
                    "away_team": "Away",
                    "result": "1",
                    "season": "2024-2025",
                    "league_name": "League",
                    "odds_1": 2.20,
                    "odds_X": 3.20,
                    "odds_2": 3.50,
                }
            ]
        )
        probabilities = pd.DataFrame({"1": [0.55], "X": [0.25], "2": [0.20]})

        bets = find_value_bets_1x2(matches, probabilities, min_edge=0.05, stake=1.0)
        summary = summarize_bets(bets)

        self.assertEqual(len(bets), 1)
        self.assertEqual(bets.loc[0, "outcome"], "1")
        self.assertAlmostEqual(summary["profit"], 1.20)
        self.assertGreater(summary["roi"], 1.0)

    def test_kelly_fraction_is_capped(self):
        self.assertAlmostEqual(kelly_fraction(0.60, 2.20, cap=0.02), 0.02)


if __name__ == "__main__":
    unittest.main()
