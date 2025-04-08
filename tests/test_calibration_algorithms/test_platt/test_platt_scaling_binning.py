import unittest
import numpy as np
from numpy.testing import assert_allclose

from Package.src.SmartCal.calibration_algorithms.platt.platt_scaling_binning import PlattBinnerScalingCalibrator


class TestPlattBinnerScalingCalibrator(unittest.TestCase):
    def setUp(self):
        """
        Setup for both binary and multiclass test cases.
        """
        self.num_bins = 5  # Number of bins for binning

        # Binary classification test data (logits)
        self.binary_logits = np.array([
            [2.0, -2.0],
            [0.5, 0.1],
            [2.0, 0.5],
            [-0.2, 0.0]
        ])
        self.binary_ground_truth = np.array([0, 1, 0, 1])

        # Multiclass classification test data (logits)
        self.multiclass_logits = np.array([
            [2.0, 1.0, 0.1],
            [1.5, 2.2, 0.3],
            [0.3, 0.5, 2.0],
            [0.8, 1.2, 1.0]
        ])
        self.multiclass_ground_truth = np.array([0, 1, 2, 1])

        # Instantiate calibrators
        self.binary_calibrator = PlattBinnerScalingCalibrator(num_bins=self.num_bins)
        self.multiclass_calibrator = PlattBinnerScalingCalibrator(num_bins=self.num_bins)

    def test_binary_calibration(self):
        """
        Test binary calibration case.
        """
        self.binary_calibrator.fit(self.binary_logits, self.binary_ground_truth)
        preds = self.binary_calibrator.predict(self.binary_logits)

        # Ensure probabilities sum to 1
        row_sums = preds.sum(axis=1)
        assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

        # Ensure shape consistency
        self.assertEqual(preds.shape, self.binary_logits.shape)

    def test_multiclass_calibration(self):
        """
        Test multiclass calibration with one-vs-all.
        """
        self.multiclass_calibrator.fit(self.multiclass_logits, self.multiclass_ground_truth)
        preds = self.multiclass_calibrator.predict(self.multiclass_logits)

        # Ensure probabilities sum to 1
        row_sums = preds.sum(axis=1)
        assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

        # Ensure shape consistency
        self.assertEqual(preds.shape, self.multiclass_logits.shape)

    def test_multiclass_predict_shape(self):
        """
        Ensure the shape of the predicted probabilities matches (n_samples, n_classes).
        """
        self.multiclass_calibrator.fit(self.multiclass_logits, self.multiclass_ground_truth)
        preds = self.multiclass_calibrator.predict(self.multiclass_logits)
        self.assertEqual(preds.shape, self.multiclass_logits.shape)

    def test_not_fitted_error(self):
        """
        Ensure predict() raises an error if fit() was not called.
        """
        with self.assertRaises(RuntimeError):
            self.binary_calibrator.predict(self.binary_logits)

    def test_invalid_input_shapes(self):
        """
        Ensure the calibrator handles invalid input shapes properly.
        """
        self.binary_calibrator.fit(self.binary_logits, self.binary_ground_truth)

        # Test incorrect shape input (e.g., wrong number of classes)
        invalid_logits = np.array([
            [1.0, 0.5, 0.2],  # 3 classes instead of 2
            [0.7, 1.2, 0.4]
        ])
        with self.assertRaises(ValueError):
            self.binary_calibrator.predict(invalid_logits)

    def test_binning_stability(self):
        """
        Ensure binning does not introduce NaN values and correctly assigns bins.
        """
        self.multiclass_calibrator.fit(self.multiclass_logits, self.multiclass_ground_truth)
        preds = self.multiclass_calibrator.predict(self.multiclass_logits)

        # Ensure no NaN values
        self.assertFalse(np.isnan(preds).any())

    def test_binning_edge_cases(self):
        """
        Ensure binning works correctly with extreme probabilities.
        """
        extreme_logits = np.array([
            [10.0, -10.0, -10.0],  # Strongly favoring class 0
            [-10.0, 10.0, -10.0],  # Strongly favoring class 1
            [-10.0, -10.0, 10.0]   # Strongly favoring class 2
        ])
        extreme_ground_truth = np.array([0, 1, 2])

        self.multiclass_calibrator.fit(extreme_logits, extreme_ground_truth)
        preds = self.multiclass_calibrator.predict(extreme_logits)

        # Ensure probabilities sum to 1
        row_sums = preds.sum(axis=1)
        assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_bin_means_storage(self):
        """
        Ensure bin means are stored correctly after fitting.
        """
        self.multiclass_calibrator.fit(self.multiclass_logits, self.multiclass_ground_truth)

        # Ensure bin_means are computed for each class
        self.assertEqual(len(self.multiclass_calibrator.bin_means), self.multiclass_logits.shape[1])

    def test_metadata_storage(self):
        """
        Ensure metadata is correctly stored after fitting.
        """
        self.multiclass_calibrator.fit(self.multiclass_logits, self.multiclass_ground_truth)
        self.multiclass_calibrator.predict(self.multiclass_logits)

        metadata = self.multiclass_calibrator.metadata
        print(metadata)

        # Check if calibration type is correctly stored
        self.assertEqual(metadata["calibration_type"], "platt_binner_scaling")

        # Ensure dataset info is correctly logged
        self.assertIn("dataset_info", metadata)
        self.assertEqual(metadata["dataset_info"]["n_samples"], len(self.multiclass_logits))
        self.assertEqual(metadata["dataset_info"]["n_classes"], self.multiclass_logits.shape[1])

        # Ensure binning parameters are stored
        self.assertIn("params", metadata)

if __name__ == '__main__':
    unittest.main()
