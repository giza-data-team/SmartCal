import unittest
import numpy as np
from numpy.testing import assert_allclose

from smartcal.calibration_algorithms.beta import BetaCalibrator


class TestBetaCalibrator(unittest.TestCase):
    def test_binary_calibration(self):
        # Synthetic binary data
        np.random.seed(0)
        n_samples = 50
        # Suppose we have uncalibrated probabilities for 2 classes
        # We'll simulate with random uniform, but ensure row sums=1
        raw_probs = np.random.rand(n_samples, 2)
        raw_probs /= raw_probs.sum(axis=1, keepdims=True)

        # Ground truth labels (0 or 1)
        labels = np.random.randint(0, 2, size=n_samples)

        # Fit calibrator
        beta_cal = BetaCalibrator(model_type="abm")
        beta_cal.fit(raw_probs, labels)

        # Predict
        calibrated = beta_cal.predict(raw_probs)

        self.assertEqual(calibrated.shape, (n_samples, 2))
        # Check row sums
        row_sums = calibrated.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-7)

    def test_multiclass_calibration(self):
        # Synthetic 3-class data
        np.random.seed(42)
        n_samples = 60
        raw_probs = np.random.rand(n_samples, 3)
        raw_probs /= raw_probs.sum(axis=1, keepdims=True)

        labels = np.random.randint(0, 3, size=n_samples)

        beta_cal = BetaCalibrator(model_type="ab")
        beta_cal.fit(raw_probs, labels)
        calibrated = beta_cal.predict(raw_probs)

        self.assertEqual(calibrated.shape, (n_samples, 3))
        # row sums => 1
        row_sums = calibrated.sum(axis=1)
        assert_allclose(row_sums, 1.0, atol=1e-7)

    def test_not_fitted_error(self):
        beta_cal = BetaCalibrator()
        raw_probs = np.array([[0.3, 0.7]])
        with self.assertRaises(RuntimeError):
            beta_cal.predict(raw_probs)


