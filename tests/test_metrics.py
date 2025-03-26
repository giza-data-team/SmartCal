import unittest
import numpy as np
from metrics.conf_ece import ConfECE
from metrics.ece import ECE
from metrics.mce import MCE
from metrics.brier_score import calculate_brier_score
from metrics.calibration_curve import calculate_calibration_curve

class TestCalibrationMetrics(unittest.TestCase):
    def setUp(self):
        self.num_bins = 10
        self.confidence_threshold = 0.5
        self.predicted_probabilities = np.array([0.6, 0.7, 0.8, 0.9, 0.4])
        self.predicted_labels = np.array([1, 1, 1, 1, 0])
        self.true_labels = np.array([1, 0, 1, 1, 0])

        self.multiclass_y_true = np.array([0, 1, 2, 1, 0])
        self.multiclass_y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.3, 0.6, 0.1],
            [0.6, 0.2, 0.2]
        ])

    def test_conf_ece(self):
        conf_ece = ConfECE(self.num_bins, self.confidence_threshold)
        conf_ece_value = conf_ece.compute(self.predicted_probabilities, self.predicted_labels, self.true_labels)
        log_data = conf_ece.logger()
        self.assertIsInstance(conf_ece_value, float)
        self.assertEqual(log_data["metric_name"], "ConfECE")
        self.assertEqual(log_data["parameters"]["num_bins"], self.num_bins)
        self.assertEqual(log_data["parameters"]["confidence_threshold"], self.confidence_threshold)

    def test_ece(self):
        ece = ECE(self.num_bins)
        ece_value = ece.compute(self.predicted_probabilities, self.predicted_labels, self.true_labels)
        log_data = ece.logger()
        self.assertIsInstance(ece_value, float)
        self.assertEqual(log_data["metric_name"], "ECE")
        self.assertEqual(log_data["parameters"]["num_bins"], self.num_bins)

    def test_mce(self):
        mce = MCE(self.num_bins)
        mce_value = mce.compute(self.predicted_probabilities, self.predicted_labels, self.true_labels)
        log_data = mce.logger()
        self.assertIsInstance(mce_value, float)
        self.assertEqual(log_data["metric_name"], "MCE")
        self.assertEqual(log_data["parameters"]["num_bins"], self.num_bins)
        
    def test_multiclass_brier_score(self):
        # Test multiclass case
        brier_score = calculate_brier_score(
            self.multiclass_y_true, 
            self.multiclass_y_prob
        )
        self.assertIsInstance(brier_score, float)
        self.assertTrue(0 <= brier_score <= 1)

        # Test binary case
        binary_brier_score = calculate_brier_score(
            self.true_labels, 
            self.predicted_probabilities
        )
        self.assertIsInstance(binary_brier_score, float)
        self.assertTrue(0 <= binary_brier_score <= 1)

        # Test perfect predictions
        perfect_y_true = np.array([0, 1])
        perfect_y_prob = np.array([[1.0, 0.0], [0.0, 1.0]])
        perfect_score = calculate_brier_score(perfect_y_true, perfect_y_prob)
        self.assertAlmostEqual(perfect_score, 0.0)

    def test_calibration_curve(self):
        # Test multiclass case
        mean_probs, true_probs, bin_counts = calculate_calibration_curve(
            self.multiclass_y_true,
            self.multiclass_y_prob,
            n_bins=self.num_bins
        )

        self.assertEqual(len(mean_probs), self.num_bins)
        self.assertEqual(len(true_probs), self.num_bins)
        self.assertEqual(len(bin_counts), self.num_bins)

        # Test binary case
        binary_mean_probs, binary_true_probs, binary_bin_counts = calculate_calibration_curve(
            self.true_labels,
            self.predicted_probabilities,
            n_bins=self.num_bins
        )

        self.assertEqual(len(binary_mean_probs), self.num_bins)
        self.assertEqual(len(binary_true_probs), self.num_bins)
        self.assertEqual(len(binary_bin_counts), self.num_bins)

    def test_edge_cases(self):
        # Test empty input
        with self.assertRaises(ValueError) as context:
            calculate_brier_score(np.array([]), np.array([]))
        self.assertEqual(str(context.exception), "Input arrays cannot be empty")

        # Test invalid number of bins
        with self.assertRaises(ValueError) as context:
            calculate_calibration_curve(
                self.multiclass_y_true,
                self.multiclass_y_prob,
                n_bins=0
            )
        self.assertEqual(str(context.exception), "Number of bins must be positive and higher than 0")


if __name__ == "__main__":
    unittest.main()
