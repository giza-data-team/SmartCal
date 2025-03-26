import unittest
import itertools
import numpy as np
from numpy.testing import assert_allclose

from calibration_algorithms.mix_and_match import MixAndMatchCalibrator
from calibration_algorithms.temperature_scaling import TemperatureScalingCalibrator
from calibration_algorithms.isotonic import IsotonicCalibrator
from calibration_algorithms.beta import BetaCalibrator
from calibration_algorithms.dirichlet import DirichletCalibrator
from calibration_algorithms.empirical_binning import EmpiricalBinningCalibrator
from calibration_algorithms.matrix_scaling import MatrixScalingCalibrator
from calibration_algorithms.adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator
from calibration_algorithms.imax import ImaxCalibrator
from calibration_algorithms.platt.platt_scaling import PlattScalingCalibrator
from calibration_algorithms.vector_scaling import VectorScalingCalibrator
from calibration_algorithms.histogram.histogram import HistogramCalibrator
from sklearn.metrics import log_loss


class TestMixAndMatchCalibrator(unittest.TestCase):
    """
    Unit tests for MixAndMatchCalibrator.
    This version dynamically tests all valid parametric + non-parametric combinations.
    """

    PARAMETRIC_CALIBRATORS = {
        "TemperatureScalingCalibrator": TemperatureScalingCalibrator,
        "PlattScalingCalibrator": PlattScalingCalibrator,
        "VectorScalingCalibrator": VectorScalingCalibrator,
        "MatrixScalingCalibrator": MatrixScalingCalibrator,
        "BetaCalibrator": BetaCalibrator,
        "DirichletCalibrator": DirichletCalibrator,
        "AdaptiveTemperatureScalingCalibrator": AdaptiveTemperatureScalingCalibrator
    }

    NONPARAMETRIC_CALIBRATORS = {
        "IsotonicCalibrator": IsotonicCalibrator,
        "EmpiricalBinningCalibrator": EmpiricalBinningCalibrator,
        "HistogramCalibrator": HistogramCalibrator,
        "ImaxCalibrator": ImaxCalibrator
    }

    def setUp(self):
        """
        Setup for both binary and multiclass test cases.
        """
        # Binary classification test data
        self.binary_probs = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.6, 0.4]
        ])
        self.binary_y = np.array([0, 1, 0, 1])

        # Multiclass data (3 classes)
        self.multi_probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.5, 0.4],
            [0.3, 0.3, 0.4],
            [0.33, 0.33, 0.34]
        ])
        self.multi_y = np.array([0, 1, 2, 2])

        # Initialize calibrator with default calibrators
        self.mix_match = MixAndMatchCalibrator()

    def test_all_combinations(self):
        """
        Test all valid parametric and non-parametric calibrator combinations.
        Ensures correctness for both binary and multiclass datasets.
        """
        for param_cal, nonparam_cal in itertools.product(
                self.PARAMETRIC_CALIBRATORS.keys(),
                self.NONPARAMETRIC_CALIBRATORS.keys()):

            with self.subTest(parametric=param_cal, nonparametric=nonparam_cal):
                calibrator = MixAndMatchCalibrator(param_cal, nonparam_cal)

                # Ensure correct class assignment
                self.assertEqual(calibrator.parametric_calibrator.__class__.__name__, param_cal)
                self.assertEqual(calibrator.nonparametric_calibrator.__class__.__name__, nonparam_cal)

                # Binary classification test
                initial_log_loss = log_loss(self.binary_y, self.binary_probs)
                calibrator.fit(self.binary_probs, self.binary_y)
                calibrated_probs = calibrator.predict(self.binary_probs)

                # Ensure shape and sum constraints hold
                self.assertEqual(calibrated_probs.shape, (4, 2))
                assert_allclose(calibrated_probs.sum(axis=1), np.ones(4), atol=1e-7)
                self.assertTrue(np.all(calibrated_probs >= 0) and np.all(calibrated_probs <= 1))

                # Ensure calibration does not worsen log loss
                calibrated_log_loss = log_loss(self.binary_y, calibrated_probs)
                self.assertLessEqual(calibrated_log_loss, initial_log_loss,
                                     f"Calibration increased log loss for {param_cal} + {nonparam_cal}")

                # Multiclass test
                initial_log_loss_multi = log_loss(self.multi_y, self.multi_probs)
                calibrator.fit(self.multi_probs, self.multi_y)
                calibrated_probs_multi = calibrator.predict(self.multi_probs)

                # Ensure shape and sum constraints hold
                self.assertEqual(calibrated_probs_multi.shape, (4, 3))
                assert_allclose(calibrated_probs_multi.sum(axis=1), np.ones(4), atol=1e-7)
                self.assertTrue(np.all(calibrated_probs_multi >= 0) and np.all(calibrated_probs_multi <= 1))

                # Ensure calibration does not worsen log loss
                calibrated_log_loss_multi = log_loss(self.multi_y, calibrated_probs_multi)
                self.assertLessEqual(calibrated_log_loss_multi, initial_log_loss_multi,
                                     f"Calibration increased log loss for {param_cal} + {nonparam_cal}")

    def test_default_initialization(self):
        """
        Test that default initialization creates correct calibrator types.
        """
        calibrator = MixAndMatchCalibrator()
        self.assertIsInstance(calibrator.parametric_calibrator, TemperatureScalingCalibrator)
        self.assertIsInstance(calibrator.nonparametric_calibrator, IsotonicCalibrator)
        self.assertFalse(calibrator.fitted)

    def test_custom_initialization(self):
        """
        Test initialization with custom calibrators.
        """
        calibrator = MixAndMatchCalibrator(parametric_calibrator="PlattScalingCalibrator", nonparametric_calibrator= "HistogramCalibrator")
        self.assertEqual(calibrator.parametric_calibrator.__class__.__name__, "PlattScalingCalibrator")
        self.assertEqual(calibrator.nonparametric_calibrator.__class__.__name__, "HistogramCalibrator")

    def test_binary_calibration(self):
        """
        Test calibration on binary classification data.
        """
        self.mix_match.fit(self.binary_probs, self.binary_y)
        preds = self.mix_match.predict(self.binary_probs)

        # Check shape and probability constraints
        self.assertEqual(preds.shape, (4, 2))
        assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-7)
        self.assertTrue(np.all(preds >= 0) and np.all(preds <= 1))

    def test_multiclass_calibration(self):
        """
        Test calibration on multiclass data.
        """
        self.mix_match.fit(self.multi_probs, self.multi_y)
        preds = self.mix_match.predict(self.multi_probs)

        # Check shape and probability constraints
        self.assertEqual(preds.shape, (4, 3))
        assert_allclose(preds.sum(axis=1), np.ones(4), atol=1e-7)
        self.assertTrue(np.all(preds >= 0) and np.all(preds <= 1))

    def test_not_fitted_error(self):
        """
        Test that predict raises an error if fit is not called first.
        """
        with self.assertRaises(RuntimeError):
            self.mix_match.predict(self.binary_probs)

    def test_calibration_effect(self):
        """
        Test that calibration improves or maintains log loss.
        """
        from sklearn.metrics import log_loss

        # Binary case
        initial_log_loss = log_loss(self.binary_y, self.binary_probs)
        self.mix_match.fit(self.binary_probs, self.binary_y)
        calibrated_probs = self.mix_match.predict(self.binary_probs)
        calibrated_log_loss = log_loss(self.binary_y, calibrated_probs)

        self.assertLessEqual(calibrated_log_loss, initial_log_loss,
                             "Calibration should not increase log loss.")

    def test_extreme_probabilities(self):
        """
        Test that the model handles extreme probabilities without numerical instability.
        """
        extreme_probs = np.array([
            [0.99, 0.01],
            [0.01, 0.99],
            [0.999, 0.001]
        ])
        y = np.array([0, 1, 0])

        self.mix_match.fit(extreme_probs, y)
        calibrated_probs = self.mix_match.predict(extreme_probs)

        assert_allclose(calibrated_probs.sum(axis=1), np.ones(3), atol=1e-7)
        self.assertTrue(np.all(calibrated_probs >= 0) and np.all(calibrated_probs <= 1))

    def test_large_dataset(self):
        """
        Test performance and correctness with a large dataset.
        """
        n_samples = 10000
        n_classes = 3
        large_probs = np.random.random((n_samples, n_classes))
        large_probs = large_probs / large_probs.sum(axis=1)[:, np.newaxis]
        large_y = np.random.randint(0, n_classes, size=n_samples)

        self.mix_match.fit(large_probs, large_y)
        calibrated_probs = self.mix_match.predict(large_probs)

        assert_allclose(calibrated_probs.sum(axis=1), np.ones(n_samples), atol=1e-7)

    def test_multiple_fit_calls(self):
        """
        Test that multiple calls to fit work as expected.
        """
        # First fit
        self.mix_match.fit(self.binary_probs, self.binary_y)
        first_preds = self.mix_match.predict(self.binary_probs)

        # Second fit with slightly modified data
        modified_probs = self.binary_probs + np.random.uniform(-0.01, 0.01, self.binary_probs.shape)
        modified_probs = modified_probs / modified_probs.sum(axis=1)[:, np.newaxis]

        self.mix_match.fit(modified_probs, self.binary_y)
        second_preds = self.mix_match.predict(self.binary_probs)

        # Predictions should be different
        self.assertFalse(np.allclose(first_preds, second_preds))

    def test_deterministic_behavior(self):
        """
        Test that the model is deterministic with the same random seed.
        """
        cal1 = MixAndMatchCalibrator(parametric_calibrator="TemperatureScalingCalibrator", nonparametric_calibrator="IsotonicCalibrator", seed=42)
        cal2 = MixAndMatchCalibrator(parametric_calibrator="AdaptiveTemperatureScalingCalibrator", nonparametric_calibrator="ImaxCalibrator", seed=42)

        cal1.fit(self.binary_probs, self.binary_y)
        cal2.fit(self.binary_probs, self.binary_y)

        pred1 = cal1.predict(self.binary_probs)
        pred2 = cal2.predict(self.binary_probs)

        assert_allclose(pred1, pred2, atol=1e-7)

    def test_invalid_combination(self):
        """
        Test that using incompatible calibrators raises an error.
        """
        with self.assertRaises(KeyError):
            MixAndMatchCalibrator(parametric_calibrator="PlattScalingCalibrator", nonparametric_calibrator="VectorScalingCalibrator")

if __name__ == '__main__':
    unittest.main()
