import unittest
import numpy as np

from calibration_algorithms.histogram.histogram import HistogramCalibrator


class TestHistogramCalibrator(unittest.TestCase):

    def setUp(self):
        """Set up test data for all test cases."""
        self.num_samples = 1000
        self.num_classes_multi = 3  # Multi-class setting
        self.num_classes_binary = 2  # Binary classification setting
        self.num_bins = 10
        self.seed = 42

        np.random.seed(self.seed)

        # Generate logits for multi-class case
        logits_multi = np.random.randn(self.num_samples, self.num_classes_multi)

        # Generate logits for binary classification case
        logits_binary = np.random.randn(self.num_samples, self.num_classes_binary)

        # Softmax function to convert logits to probabilities
        def softmax(logits):
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.probabilities_multi = softmax(logits_multi)
        self.probabilities_binary = softmax(logits_binary)

        # Generate synthetic ground truth labels
        self.ground_truth_multi = np.random.randint(0, self.num_classes_multi, size=self.num_samples)
        self.ground_truth_binary = np.random.randint(0, self.num_classes_binary, size=self.num_samples)

        # Available Histogram calibrator types
        self.histogram_types = ["HISTOGRAM", "HISTOGRAMMARGINAL"]

    def test_initialization(self):
        """Test that the calibrator initializes correctly for all types."""
        for hist_type in self.histogram_types:
            with self.subTest(hist_type=hist_type):
                calibrator = HistogramCalibrator(calibrator_type=hist_type, num_bins=self.num_bins)
                self.assertEqual(calibrator.calibrator_type.name, hist_type)
                self.assertEqual(calibrator.num_bins, self.num_bins)
                self.assertFalse(calibrator.fitted)

    def test_metadata_after_fit(self):
        """Test if metadata is updated correctly after fitting."""
        calibrator = HistogramCalibrator(calibrator_type="HISTOGRAM", num_bins=self.num_bins)
        calibrator.fit(self.probabilities_multi, self.ground_truth_multi)

        metadata = calibrator.metadata
        self.assertEqual(metadata["dataset_info"]["n_samples"], self.num_samples)
        self.assertEqual(metadata["dataset_info"]["n_classes"], self.num_classes_multi)
        self.assertEqual(metadata["params"]["Histogram Calibration Type"], "HISTOGRAM")
        self.assertEqual(metadata["params"]["num_bins"], self.num_bins)

    def test_binary_calibration(self):
        """Test calibration on a binary classification dataset."""
        calibrator = HistogramCalibrator(calibrator_type="HISTOGRAM", num_bins=self.num_bins)
        calibrator.fit(self.probabilities_binary, self.ground_truth_binary)

        test_logits = np.random.randn(10, self.num_classes_binary)
        test_probabilities = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)

        calibrated_probs = calibrator.predict(test_probabilities)
        self.assertEqual(calibrated_probs.shape, test_probabilities.shape)
        np.testing.assert_almost_equal(calibrated_probs.sum(axis=1), np.ones(10), decimal=5)

    def test_multi_class_calibration(self):
        """Test calibration on a multi-class dataset."""
        calibrator = HistogramCalibrator(calibrator_type="HISTOGRAM", num_bins=self.num_bins)
        calibrator.fit(self.probabilities_multi, self.ground_truth_multi)

        test_logits = np.random.randn(10, self.num_classes_multi)
        test_probabilities = np.exp(test_logits) / np.sum(np.exp(test_logits), axis=1, keepdims=True)

        calibrated_probs = calibrator.predict(test_probabilities)
        self.assertEqual(calibrated_probs.shape, test_probabilities.shape)
        np.testing.assert_almost_equal(calibrated_probs.sum(axis=1), np.ones(10), decimal=5)

    def test_predict_without_fit(self):
        """Test that predict() raises an error if called before fit()."""
        calibrator = HistogramCalibrator(calibrator_type="HISTOGRAM", num_bins=self.num_bins)
        test_data = np.random.rand(10, self.num_classes_multi)

        with self.assertRaises(RuntimeError):
            calibrator.predict(test_data)

    def test_invalid_input_shapes(self):
        """Test that invalid input shapes raise errors."""
        calibrator = HistogramCalibrator(calibrator_type="HISTOGRAM", num_bins=self.num_bins)
        calibrator.fit(self.probabilities_multi, self.ground_truth_multi)

        invalid_data = np.array([0.1, 0.2, 0.7])  # Should be 2D
        with self.assertRaises(ValueError):
            calibrator.predict(invalid_data)

        invalid_data = np.random.rand(10, 5)  # Wrong number of columns
        with self.assertRaises(ValueError):
            calibrator.predict(invalid_data)

if __name__ == '__main__':
    unittest.main()
