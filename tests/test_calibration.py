import unittest

import pandas as pd

from soccer_predictor.calibration import blend_probabilities, fit_blend_alpha


class CalibrationTests(unittest.TestCase):
    def test_blend_probabilities_normalizes_rows(self):
        model = pd.DataFrame({"1": [0.6], "X": [0.2], "2": [0.2]})
        anchor = pd.DataFrame({"1": [0.4], "X": [0.3], "2": [0.3]})

        blended = blend_probabilities(model, anchor, alpha=0.25, labels=("1", "X", "2"))

        self.assertAlmostEqual(blended.loc[0].sum(), 1.0)
        self.assertAlmostEqual(blended.loc[0, "1"], 0.45)

    def test_fit_blend_alpha_picks_better_candidate(self):
        actual = pd.Series(["1", "1", "2"])
        model = pd.DataFrame({"1": [0.8, 0.7, 0.2], "X": [0.1, 0.2, 0.2], "2": [0.1, 0.1, 0.6]})
        anchor = pd.DataFrame({"1": [0.4, 0.3, 0.6], "X": [0.3, 0.3, 0.2], "2": [0.3, 0.4, 0.2]})

        alpha, search = fit_blend_alpha(
            actual,
            model,
            anchor,
            labels=("1", "X", "2"),
            candidates=(0.0, 1.0),
        )

        self.assertEqual(alpha, 1.0)
        self.assertEqual(len(search), 2)


if __name__ == "__main__":
    unittest.main()
