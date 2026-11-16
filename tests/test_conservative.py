import unittest

import pandas as pd

from soccer_predictor.conservative import (
    fit_market_anchor_alphas,
    market_probability_metrics,
    score_market_anchored_candidates,
    select_conservative_v2_bets,
)


class ConservativeV2Tests(unittest.TestCase):
    def test_market_anchor_normalizes_each_match_market(self):
        candidates = _candidate_frame()

        scored = score_market_anchored_candidates(
            candidates,
            {"1x2": 0.5, "over_under_2_5": 0.25},
        )

        sums = scored.groupby(["match_id", "market"])["conservative_probability"].sum()
        self.assertTrue((sums.round(10) == 1.0).all())

    def test_anchor_fit_prefers_bookmaker_when_residual_is_wrong(self):
        candidates = _candidate_frame()
        candidates["value_probability"] = [0.1, 0.1, 0.8, 0.1, 0.9]
        candidates["bookmaker_probability"] = [0.7, 0.2, 0.1, 0.8, 0.2]

        alphas, search = fit_market_anchor_alphas(
            candidates,
            alpha_candidates=(0.0, 0.5),
        )

        self.assertEqual(alphas["1x2"], 0.0)
        self.assertEqual(alphas["over_under_2_5"], 0.0)
        self.assertEqual(len(search), 4)

    def test_v2_keeps_separate_market_weekend_budgets(self):
        candidates = _candidate_frame()
        scored = score_market_anchored_candidates(
            candidates,
            {"1x2": 1.0, "over_under_2_5": 1.0},
        )

        selected = select_conservative_v2_bets(
            scored,
            {"1x2": -1.0, "over_under_2_5": -1.0},
            min_probability=0.0,
            max_bets_per_market_weekend=1,
        )

        self.assertEqual(set(selected["market"]), {"1x2", "over_under_2_5"})
        self.assertTrue((selected["selection_mode"] == "conservative_v2").all())

    def test_probability_metrics_reward_correct_predictions(self):
        candidates = _candidate_frame()

        metrics = market_probability_metrics(
            candidates[candidates["market"] == "1x2"],
            "bookmaker_probability",
        )

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertGreater(metrics["log_loss"], 0.0)


def _candidate_frame():
    rows = [
        _candidate(1, "1x2", "1", True, 0.60, 0.70, 1.5),
        _candidate(1, "1x2", "X", False, 0.25, 0.20, 4.0),
        _candidate(1, "1x2", "2", False, 0.15, 0.10, 8.0),
        _candidate(2, "over_under_2_5", "+", True, 0.65, 0.60, 1.8),
        _candidate(2, "over_under_2_5", "-", False, 0.35, 0.40, 2.2),
    ]
    return pd.DataFrame(rows)


def _candidate(match_id, market, outcome, won, value_probability, bookmaker_probability, odds):
    return {
        "match_id": match_id,
        "date": pd.Timestamp("2024-01-06"),
        "market": market,
        "outcome": outcome,
        "won": won,
        "profit": odds - 1.0 if won else -1.0,
        "decimal_odds": odds,
        "clv_decimal": 0.01,
        "clv_probability": 0.01,
        "value_probability": value_probability,
        "value_edge": value_probability * odds - 1.0,
        "bookmaker_probability": bookmaker_probability,
    }


if __name__ == "__main__":
    unittest.main()
