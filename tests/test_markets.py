import unittest

from soccer_predictor.markets import (
    both_teams_to_score,
    double_chance,
    expected_value,
    match_result_and_over_under,
    negative_binomial_score_matrix,
    over_under,
    remove_vig,
    score_matrix,
    three_way,
)


class MarketTests(unittest.TestCase):
    def test_score_matrix_is_normalized(self):
        table = score_matrix(1.6, 1.1)

        self.assertAlmostEqual(table.total_probability, 1.0, places=12)

    def test_three_way_probabilities_sum_to_one(self):
        table = score_matrix(1.6, 1.1)
        result = three_way(table)

        self.assertAlmostEqual(sum(result.values()), 1.0, places=12)
        self.assertGreater(result["1"], result["2"])

    def test_equal_expected_goals_are_symmetric(self):
        table = score_matrix(1.25, 1.25)
        result = three_way(table)

        self.assertAlmostEqual(result["1"], result["2"], places=12)

    def test_double_chance_uses_three_way_probabilities(self):
        table = score_matrix(1.6, 1.1)
        result = three_way(table)
        dc = double_chance(table)

        self.assertAlmostEqual(dc["1X"], result["1"] + result["X"], places=12)
        self.assertAlmostEqual(dc["12"], result["1"] + result["2"], places=12)
        self.assertAlmostEqual(dc["X2"], result["X"] + result["2"], places=12)

    def test_over_under_half_goal_line_has_no_push(self):
        table = score_matrix(1.6, 1.1)
        result = over_under(table, 2.5)

        self.assertAlmostEqual(result["+"] + result["-"], 1.0, places=12)
        self.assertEqual(result["push"], 0.0)

    def test_both_teams_to_score_sums_to_one(self):
        table = score_matrix(1.6, 1.1)
        result = both_teams_to_score(table)

        self.assertAlmostEqual(result["Y"] + result["N"], 1.0, places=12)

    def test_match_result_and_over_under_sums_to_one_for_half_line(self):
        table = score_matrix(1.6, 1.1)
        result = match_result_and_over_under(table, 1.5)

        self.assertAlmostEqual(sum(result.values()), 1.0, places=12)

    def test_remove_vig_normalizes_bookmaker_probabilities(self):
        probabilities = remove_vig({"1": 1.67, "X": 3.70, "2": 5.50})

        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=12)

    def test_expected_value(self):
        self.assertAlmostEqual(expected_value(0.60, 2.00), 0.20)
        self.assertAlmostEqual(expected_value(0.50, 2.00), 0.00)

    def test_dixon_coles_negative_rho_increases_draw_probability(self):
        independent = three_way(score_matrix(1.2, 1.2))
        adjusted = three_way(score_matrix(1.2, 1.2, dixon_coles_rho=-0.10))

        self.assertGreater(adjusted["X"], independent["X"])

    def test_dixon_coles_zero_rho_matches_independent_poisson(self):
        independent = three_way(score_matrix(1.4, 0.9))
        adjusted = three_way(score_matrix(1.4, 0.9, dixon_coles_rho=0.0))

        self.assertAlmostEqual(adjusted["1"], independent["1"], places=12)
        self.assertAlmostEqual(adjusted["X"], independent["X"], places=12)
        self.assertAlmostEqual(adjusted["2"], independent["2"], places=12)

    def test_zero_dispersion_matches_independent_poisson(self):
        independent = score_matrix(1.7, 1.2, max_goals=15)
        overdispersed = negative_binomial_score_matrix(
            1.7,
            1.2,
            dispersion=0.0,
            max_goals=15,
        )

        for home_goals, away_goals, probability in independent.scores():
            self.assertAlmostEqual(
                overdispersed.probability(home_goals, away_goals),
                probability,
                places=12,
            )

    def test_negative_binomial_has_a_heavier_total_goals_tail(self):
        independent = score_matrix(1.7, 1.2, max_goals=15)
        overdispersed = negative_binomial_score_matrix(
            1.7,
            1.2,
            dispersion=0.25,
            max_goals=15,
        )

        self.assertAlmostEqual(overdispersed.total_probability, 1.0, places=12)
        self.assertGreater(
            over_under(overdispersed, 5.5)["+"],
            over_under(independent, 5.5)["+"],
        )


if __name__ == "__main__":
    unittest.main()
