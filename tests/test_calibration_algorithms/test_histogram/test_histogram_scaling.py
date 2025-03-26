import unittest
import numpy as np
from numpy.testing import assert_allclose

from calibration_algorithms.histogram.histogram_scaling import HistogramScalingCalibrator  

class TestHistogramScalingCalibrator(unittest.TestCase):
    def setUp(self):
        """
        Setup test cases for binary and multi-class calibration.
        """
        np.random.seed(42)
        self.num_bins = 10
        self.num_samples = 100
        self.num_classes_binary = 2
        self.num_classes_multi = 3
        self.num_classes_list = [3, 5, 7]  # Additional multi-class cases

        # Generate synthetic logits (binary classification)
        self.binary_logits = np.random.rand(self.num_samples, self.num_classes_binary)
        self.binary_ground_truth = np.random.randint(0, self.num_classes_binary, size=self.num_samples)

        # Generate synthetic logits (multi-class classification)
        self.multi_logits = np.random.rand(self.num_samples, self.num_classes_multi)
        self.multi_ground_truth = np.random.randint(0, self.num_classes_multi, size=self.num_samples)

        # Generate multi-class logits for various class sizes
        self.multi_logits_dict = {
            num_classes: np.random.rand(self.num_samples, num_classes) for num_classes in self.num_classes_list
        }
        self.multi_ground_truth_dict = {
            num_classes: np.random.randint(0, num_classes, size=self.num_samples) for num_classes in self.num_classes_list
        }

    def test_binary_calibration(self):
        """
        Ensure HistogramCalibrator works for binary classification.
        """
        calibrator = HistogramScalingCalibrator(num_bins=self.num_bins)
        calibrator.fit(self.binary_logits, self.binary_ground_truth)

        # Predict calibrated probabilities
        calibrated_probs = calibrator.predict(self.binary_logits)
        print(calibrator.metadata)

        # Ensure shape consistency
        self.assertEqual(calibrated_probs.shape, self.binary_logits.shape)

        # Ensure probabilities sum to 1
        row_sums = calibrated_probs.sum(axis=1)
        assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_multiclass_calibration(self):
        """
        Ensure HistogramCalibrator works for multi-class classification.
        """
        calibrator = HistogramScalingCalibrator(num_bins=self.num_bins)
        calibrator.fit(self.multi_logits, self.multi_ground_truth)

        # Predict calibrated probabilities
        calibrated_probs = calibrator.predict(self.multi_logits)

        # Ensure shape consistency
        self.assertEqual(calibrated_probs.shape, self.multi_logits.shape)

        # Ensure probabilities sum to 1
        row_sums = calibrated_probs.sum(axis=1)
        assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_multiclass_varied_classes(self):
        """
        Ensure HistogramCalibrator works for multiple class sizes (3, 5, 7).
        """
        for num_classes in self.num_classes_list:
            with self.subTest(num_classes=num_classes):
                calibrator = HistogramScalingCalibrator(num_bins=self.num_bins)
                logits = self.multi_logits_dict[num_classes]
                ground_truth = self.multi_ground_truth_dict[num_classes]

                # Fit calibrator
                calibrator.fit(logits, ground_truth)

                # Predict calibrated probabilities
                calibrated_probs = calibrator.predict(logits)

                print(f"Multi-class ({num_classes}) Calibrated Probabilities:\n", calibrated_probs)
                print(f"Multi-class ({num_classes}) Metadata:\n", calibrator.metadata)

                # Ensure shape consistency
                self.assertEqual(calibrated_probs.shape, logits.shape)

                # Ensure probabilities sum to 1
                row_sums = calibrated_probs.sum(axis=1)
                assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_invalid_input_shapes(self):
        """
        Ensure the calibrator handles invalid input shapes properly.
        """
        calibrator = HistogramScalingCalibrator(num_bins=self.num_bins)
        calibrator.fit(self.multi_logits, self.multi_ground_truth)

        # Incorrect shape input (e.g., wrong number of classes)
        invalid_logits = np.random.rand(self.num_samples, 5)  # 5 classes instead of 3
        with self.assertRaises(ValueError):
            calibrator.predict(invalid_logits)

    def test_not_fitted_error(self):
        """
        Ensure predict() raises an error if fit() was not called.
        """
        calibrator = HistogramScalingCalibrator(num_bins=self.num_bins)
        with self.assertRaises(RuntimeError):
            calibrator.predict(self.multi_logits)

    def test_metadata_storage(self):
        """
        Ensure metadata is correctly stored after fitting.
        """
        calibrator = HistogramScalingCalibrator(num_bins=self.num_bins)
        calibrator.fit(self.multi_logits, self.multi_ground_truth)
        
        metadata = calibrator.metadata

        # Ensure dataset info is correctly stored
        self.assertIn("dataset_info", metadata)
        self.assertEqual(metadata["dataset_info"]["n_samples"], self.num_samples)
        self.assertEqual(metadata["dataset_info"]["n_classes"], self.num_classes_multi)

        # Ensure bin edges are stored
        self.assertIn("params", metadata)

    def test_metadata_storage_multiclass(self):
        """
        Ensure metadata is correctly stored for multiple class sizes.
        """
        for num_classes in self.num_classes_list:
            with self.subTest(num_classes=num_classes):
                calibrator = HistogramScalingCalibrator(num_bins=self.num_bins)
                calibrator.fit(self.multi_logits_dict[num_classes], self.multi_ground_truth_dict[num_classes])
                
                metadata = calibrator.metadata

                # Ensure dataset info is correctly stored
                self.assertIn("dataset_info", metadata)
                self.assertEqual(metadata["dataset_info"]["n_samples"], self.num_samples)
                self.assertEqual(metadata["dataset_info"]["n_classes"], num_classes)

                # Ensure bin edges are stored
                self.assertIn("params", metadata)
                self.assertIn("num_bins", metadata["params"])

if __name__ == '__main__':
    unittest.main()
