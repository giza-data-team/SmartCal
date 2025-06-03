import unittest
import numpy as np
from numpy.testing import assert_allclose

from smartcal.calibration_algorithms.platt.platt_scaling import PlattScalingCalibrator


class TestPlattScalingCalibrator(unittest.TestCase):
    def setUp(self):
        """
        Setup for both binary and multiclass test cases.
        """
        # Binary classification test data
        self.binary_predictions = np.array([
            [2.0, -2.0],
            [0.5,  0.1],
            [2.0,  0.5],
            [-0.2,  0.0]
        ])
        self.binary_ground_truth = np.array([0, 1, 0, 1])
        self.binary_expected_probs = np.array([
            [0.483667, 0.516333],
            [0.51526, 0.48474],
            [0.517405, 0.482595],
            [0.483667, 0.516333]
        ])

        # Multiclass classification test data
        self.multiclass_predictions = np.array([
            [2.0, 1.0, 0.1],
            [1.5, 2.2, 0.3],
            [0.3, 0.5, 2.0],
            [0.8, 1.2, 1.0]
        ])
        self.multiclass_ground_truth = np.array([0, 1, 2, 1])
        self.multiclass_expected_probs = np.array([
            [0.47619, 0.47619, 0.047619],
            [0.434783, 0.434783, 0.130435],
            [0.166667, 0.277778, 0.555556],
            [0.285714, 0.357143, 0.357143]
        ])

        # Instantiate calibrators
        self.binary_calibrator = PlattScalingCalibrator()
        self.multiclass_calibrator = PlattScalingCalibrator()

    def test_binary_calibration(self):
        """
        Test binary calibration case.
        """
        self.binary_calibrator.fit(self.binary_predictions, self.binary_ground_truth)
        preds = self.binary_calibrator.predict(self.binary_predictions)
        assert_allclose(preds, self.binary_expected_probs, rtol=1e-3, atol=1e-3)

    def test_binary_probability_sums(self):
        """
        Ensure probabilities sum to 1 in binary case.
        """
        self.binary_calibrator.fit(self.binary_predictions, self.binary_ground_truth)
        preds = self.binary_calibrator.predict(self.binary_predictions)
        row_sums = preds.sum(axis=1)
        assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_multiclass_calibration(self):
        """
        Test multiclass calibration with one-vs-all.
        """
        self.multiclass_calibrator.fit(self.multiclass_predictions, self.multiclass_ground_truth)
        preds = self.multiclass_calibrator.predict(self.multiclass_predictions)
        assert_allclose(preds, self.multiclass_expected_probs, rtol=1e-3, atol=1e-3)

    def test_multiclass_probability_sums(self):
        """
        Ensure probabilities sum to 1 in multiclass case.
        """
        self.multiclass_calibrator.fit(self.multiclass_predictions, self.multiclass_ground_truth)
        preds = self.multiclass_calibrator.predict(self.multiclass_predictions)
        row_sums = preds.sum(axis=1)
        assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_multiclass_predict_shape(self):
        """
        Ensure the shape of the predicted probabilities matches (n_samples, n_classes).
        """
        self.multiclass_calibrator.fit(self.multiclass_predictions, self.multiclass_ground_truth)
        preds = self.multiclass_calibrator.predict(self.multiclass_predictions)
        self.assertEqual(preds.shape, self.multiclass_predictions.shape)

    def test_not_fitted_error(self):
        """
        Ensure predict() raises an error if fit() was not called.
        """
        with self.assertRaises(RuntimeError):
            self.binary_calibrator.predict(self.binary_predictions)

if __name__ == '__main__':
    unittest.main()
