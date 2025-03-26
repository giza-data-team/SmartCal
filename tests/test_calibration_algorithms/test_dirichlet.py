import unittest
import numpy as np
import torch
from calibration_algorithms.dirichlet import DirichletCalibrator

class TestDirichletCalibrator(unittest.TestCase):

    def setUp(self):
        """Set up test data for unit tests"""
        self.num_classes = 3
        self.num_samples = 100
        self.lr = 0.01
        self.max_iter = 50

        # Generate random logits and ground-truth labels
        np.random.seed(42)
        self.logits = np.random.randn(self.num_samples, self.num_classes).astype(np.float32)
        self.labels = np.random.randint(0, self.num_classes, size=self.num_samples)

        # Initialize calibrator
        self.calibrator = DirichletCalibrator(lr=self.lr, max_iter=self.max_iter)

    def test_initialization(self):
        """Test if the calibrator initializes with correct default values."""
        self.assertEqual(self.calibrator.lr, self.lr)
        self.assertEqual(self.calibrator.max_iter, self.max_iter)
        self.assertIsNone(self.calibrator.transformation_matrix)
        self.assertIsNone(self.calibrator.bias)

    def test_fitting_process(self):
        """Test if the model can fit to provided data without errors."""
        self.calibrator.fit(self.logits, self.labels)
        
        # Check if model has been marked as fitted
        self.assertTrue(self.calibrator.fitted)

        # Ensure transformation matrix and bias are properly initialized
        self.assertIsNotNone(self.calibrator.transformation_matrix)
        self.assertIsNotNone(self.calibrator.bias)

        # Check transformation matrix dimensions
        self.assertEqual(self.calibrator.transformation_matrix.shape, (self.num_classes, self.num_classes))
        self.assertEqual(self.calibrator.bias.shape, (self.num_classes,))

    def test_predict_without_fitting(self):
        """Test if an error is raised when predict() is called before fit()."""
        new_logits = np.random.randn(10, self.num_classes).astype(np.float32)
        with self.assertRaises(RuntimeError):
            self.calibrator.predict(new_logits)

    def test_valid_predictions(self):
        """Test if predict() outputs valid probability distributions."""
        self.calibrator.fit(self.logits, self.labels)

        new_logits = np.random.randn(10, self.num_classes).astype(np.float32)
        calibrated_probs = self.calibrator.predict(new_logits)

        # Check shape consistency
        self.assertEqual(calibrated_probs.shape, (10, self.num_classes))

        # Ensure probabilities sum to 1
        np.testing.assert_almost_equal(calibrated_probs.sum(axis=1), np.ones(10), decimal=5)

        # Ensure probabilities are within valid range [0,1]
        self.assertTrue((calibrated_probs >= 0).all() and (calibrated_probs <= 1).all())

    def test_invalid_inputs(self):
        """Test if appropriate errors are raised for invalid inputs."""
        self.calibrator.fit(self.logits, self.labels)

        # Non-array input
        with self.assertRaises(ValueError):
            self.calibrator.predict("invalid_input")

        # Wrong shape input (1D array)
        with self.assertRaises(ValueError):
            self.calibrator.predict(np.random.randn(10).astype(np.float32))

        # Mismatched number of classes
        with self.assertRaises(ValueError) as cm:
            self.calibrator.predict(np.random.randn(10, 5).astype(np.float32))  # 5 classes instead of 3
        self.assertIn("Mismatch in number of classes", str(cm.exception))  # Ensure correct error message

if __name__ == "__main__":
    unittest.main()
