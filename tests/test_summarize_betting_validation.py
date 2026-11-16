import unittest

import pandas as pd

from scripts.summarize_betting_validation import aggregate_results
from scripts.summarize_conservative_v2 import (
    build_comparison_frame,
    normalize_v2_record,
)


class BettingValidationSummaryTests(unittest.TestCase):
    def test_aggregate_results_weights_bets_and_clv_counts(self):
        results = pd.DataFrame(
            [
                {
                    "selection_mode": "value",
                    "status": "active",
                    "bets": 10,
                    "total_staked": 10.0,
                    "profit": 2.0,
                    "wins": 5,
                    "avg_odds": 2.0,
                    "with_closing_odds": 10,
                    "positive_clv_bets": 6,
                    "avg_decimal_clv": 0.02,
                },
                {
                    "selection_mode": "value",
                    "status": "active",
                    "bets": 20,
                    "total_staked": 20.0,
                    "profit": -1.0,
                    "wins": 8,
                    "avg_odds": 3.0,
                    "with_closing_odds": 20,
                    "positive_clv_bets": 10,
                    "avg_decimal_clv": 0.01,
                },
            ]
        )

        summary = aggregate_results(results).iloc[0]

        self.assertEqual(summary["bets"], 30)
        self.assertAlmostEqual(summary["roi"], 1.0 / 30.0)
        self.assertAlmostEqual(summary["hit_rate"], 13.0 / 30.0)
        self.assertAlmostEqual(summary["positive_clv_rate"], 16.0 / 30.0)
        self.assertAlmostEqual(summary["avg_odds"], 8.0 / 3.0)

    def test_v2_status_requires_an_active_market_threshold(self):
        base = {
            "test_season": 2025,
            "bets": 0,
            "total_staked": 0.0,
            "profit": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "avg_odds": 0.0,
            "with_closing_odds": 0,
            "positive_clv_rate": 0.0,
            "avg_decimal_clv": 0.0,
            "alpha_by_market": {"1x2": 0.5, "over_under_2_5": 0.5},
        }

        abstained = normalize_v2_record(
            {**base, "thresholds_by_market": {"1x2": None, "over_under_2_5": None}}
        )
        active = normalize_v2_record(
            {**base, "thresholds_by_market": {"1x2": 0.01, "over_under_2_5": None}}
        )

        self.assertEqual(abstained["status"], "abstained")
        self.assertEqual(active["status"], "active")

    def test_comparison_frame_labels_both_versions(self):
        common = {
            "test_season": 2022,
            "bets": 10,
            "total_staked": 10.0,
            "profit": 1.0,
            "roi": 0.1,
            "hit_rate": 0.6,
            "avg_odds": 1.7,
            "with_closing_odds": 10,
            "positive_clv_rate": 0.5,
            "avg_decimal_clv": 0.01,
        }
        frame = build_comparison_frame(
            [{**common, "chosen_edge_threshold": 0.02}],
            [
                {
                    **common,
                    "alpha_by_market": {"1x2": 0.3},
                    "thresholds_by_market": {"1x2": 0.01},
                }
            ],
        )

        self.assertEqual(
            set(frame["selection_mode"]),
            {"conservative_v1", "conservative_v2"},
        )


if __name__ == "__main__":
    unittest.main()
