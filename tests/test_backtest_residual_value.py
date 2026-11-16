import unittest

import pandas as pd

from scripts.backtest_residual_value import split_season_data


class ResidualBacktestScopeTests(unittest.TestCase):
    def test_split_season_data_supports_unseen_league_holdout(self):
        data = pd.DataFrame(
            [
                {"season_start_year": 2023, "league_code": "A"},
                {"season_start_year": 2023, "league_code": "B"},
                {"season_start_year": 2024, "league_code": "A"},
                {"season_start_year": 2024, "league_code": "B"},
            ]
        )

        train, test = split_season_data(
            data,
            season=2024,
            train_league_codes={"A"},
            test_league_codes={"B"},
        )

        self.assertEqual(set(train["league_code"]), {"A"})
        self.assertEqual(set(test["league_code"]), {"B"})
        self.assertTrue((train["season_start_year"] < 2024).all())
        self.assertTrue((test["season_start_year"] == 2024).all())


if __name__ == "__main__":
    unittest.main()
