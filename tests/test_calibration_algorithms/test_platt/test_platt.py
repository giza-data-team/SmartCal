import unittest
import numpy as np
from numpy.testing import assert_allclose

from smartcal.calibration_algorithms.platt.platt import PlattCalibrator


class TestPlattCalibrator(unittest.TestCase):
    def setUp(self):
        """
        Setup test cases for both binary and multi-class classification.
        """
        np.random.seed(42)
        self.num_bins = 10
        self.num_samples = 100
        self.num_classes_binary = 2
        self.num_classes_multi = 3

        # Generate synthetic logits (binary classification)
        self.binary_logits = np.random.randn(self.num_samples, self.num_classes_binary) * 3
        self.binary_ground_truth = np.random.randint(0, self.num_classes_binary, size=self.num_samples)

        # Generate synthetic logits (multi-class classification)
        self.multi_logits = np.random.randn(self.num_samples, self.num_classes_multi) * 3
        self.multi_ground_truth = np.random.randint(0, self.num_classes_multi, size=self.num_samples)

        # Available Platt calibrator types
        self.platt_types = ["PLATT", "PLATTBINNER", "PLATTBINNERMARGINAL"]

    def test_binary_calibration(self):
        """
        Ensure PlattCalibrator works for binary classification.
        """
        for platt_type in self.platt_types:
            with self.subTest(platt_type=platt_type):
                calibrator = PlattCalibrator(calibrator_type=platt_type, num_bins=self.num_bins)

                # Fit calibrator
                calibrator.fit(self.binary_logits, self.binary_ground_truth)

                # Predict calibrated probabilities
                calibrated_probs = calibrator.predict(self.binary_logits)

                # Ensure shape consistency
                self.assertEqual(calibrated_probs.shape, self.binary_logits.shape)

                # Ensure probabilities sum to 1
                row_sums = calibrated_probs.sum(axis=1)
                assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_multiclass_calibration(self):
        """
        Ensure PlattCalibrator works for multi-class classification.
        """
        for platt_type in self.platt_types:
            with self.subTest(platt_type=platt_type):
                calibrator = PlattCalibrator(calibrator_type=platt_type, num_bins=self.num_bins)

                # Fit calibrator
                calibrator.fit(self.multi_logits, self.multi_ground_truth)

                # Predict calibrated probabilities
                calibrated_probs = calibrator.predict(self.multi_logits)

                # Ensure shape consistency
                self.assertEqual(calibrated_probs.shape, self.multi_logits.shape)

                # Ensure probabilities sum to 1
                row_sums = calibrated_probs.sum(axis=1)
                assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-6)

    def test_invalid_input_shapes(self):
        """
        Ensure the calibrator handles invalid input shapes properly.
        """
        calibrator = PlattCalibrator(calibrator_type="PLATT", num_bins=self.num_bins)
        calibrator.fit(self.multi_logits, self.multi_ground_truth)

        # Incorrect shape input (e.g., wrong number of classes)
        invalid_logits = np.random.randn(self.num_samples, 5)  # 5 classes instead of 3
        with self.assertRaises(ValueError):
            calibrator.predict(invalid_logits)

    def test_not_fitted_error(self):
        """
        Ensure predict() raises an error if fit() was not called.
        """
        for platt_type in self.platt_types:
            with self.subTest(platt_type=platt_type):
                calibrator = PlattCalibrator(calibrator_type=platt_type, num_bins=self.num_bins)
                with self.assertRaises(RuntimeError):
                    calibrator.predict(self.multi_logits)

    def test_metadata_storage(self):
        """
        Ensure metadata is correctly stored after fitting.
        """
        for platt_type in self.platt_types:
            with self.subTest(platt_type=platt_type):
                calibrator = PlattCalibrator(calibrator_type=platt_type, num_bins=self.num_bins)
                calibrator.fit(self.multi_logits, self.multi_ground_truth)
                calibrator.predict(self.multi_logits)
                
                metadata = calibrator.metadata
                print(metadata)

                # Ensure dataset info is correctly stored
                self.assertIn("dataset_info", metadata)
                self.assertEqual(metadata["dataset_info"]["n_samples"], self.num_samples)
                self.assertEqual(metadata["dataset_info"]["n_classes"], self.num_classes_multi)

                # Ensure Platt parameters are stored
                self.assertIn("params", metadata)
                self.assertIn("Platt Calibration Type", metadata["params"])
                self.assertEqual(metadata["params"]["Platt Calibration Type"], platt_type)
                self.assertEqual(metadata["params"]["num_bins"], self.num_bins)

if __name__ == '__main__':
    unittest.main()
