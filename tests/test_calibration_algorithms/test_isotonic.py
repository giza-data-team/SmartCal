import unittest
import numpy as np
from numpy.testing import assert_allclose

from smartcal.calibration_algorithms.isotonic import IsotonicCalibrator


class TestIsotonicCalibrator(unittest.TestCase):
    def setUp(self):
        """
        Setup for both binary and multiclass test cases.
        We'll reuse or create small synthetic data sets.
        """
        # Binary classification test data:
        # We'll treat these as "probabilities" (not logits)
        self.binary_probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.6, 0.4]
        ])
        # Ground truth: class 0 or 1
        self.binary_y = np.array([0, 1, 0, 1])

        # For the sake of demonstration, let's define some "expected" values
        # that we might get from an isotonic calibration.
        # In practice, you might just check shape and sums,
        # or compare to a known reference if you have one.
        self.binary_expected = np.array([
            [1., 0.],
            [0., 1.],
            [1., 0.],
            [0., 1.]
        ])

        # Multiclass data (3 classes):
        self.multi_probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.5, 0.4],
            [0.3, 0.3, 0.4],
            [0.33, 0.33, 0.34]
        ])
        self.multi_y = np.array([0, 1, 2, 2])

        # Just a placeholder "expected" we might see
        self.multi_expected = np.array([
            [1., 0., 0.],
            [0., 0.6, 0.4],
            [0., 0., 1.],
            [0., 0., 1.]
        ])

        self.iso_bin = IsotonicCalibrator()
        self.iso_multi = IsotonicCalibrator()

    def test_binary_calibration(self):
        self.iso_bin.fit(self.binary_probs, self.binary_y)
        preds = self.iso_bin.predict(self.binary_probs)
        # Check shape
        self.assertEqual(preds.shape, (4, 2))
        # Check row sums
        assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-7)
        # Optionally compare to a "reference"
        # In reality isotonic can produce slightly different results
        # depending on the data. Let's just use rtol=0.2 for a loose check
        assert_allclose(preds, self.binary_expected, rtol=0.2, atol=0.2)

    def test_binary_not_fitted(self):
        iso_new = IsotonicCalibrator()
        with self.assertRaises(RuntimeError):
            iso_new.predict(self.binary_probs)

    def test_multiclass_calibration(self):
        self.iso_multi.fit(self.multi_probs, self.multi_y)
        preds = self.iso_multi.predict(self.multi_probs)
        self.assertEqual(preds.shape, (4, 3))
        # row sums should be ~1
        assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-7)
        # Compare to the "expected" placeholder
        assert_allclose(preds, self.multi_expected, rtol=0.2, atol=0.2)

    def test_multiclass_not_fitted(self):
        iso_new = IsotonicCalibrator()
        with self.assertRaises(RuntimeError):
            iso_new.predict(self.multi_probs)
