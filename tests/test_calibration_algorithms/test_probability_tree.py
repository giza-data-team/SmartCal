import unittest
import numpy as np
import torch
from scipy.special import softmax
from Package.src.SmartCal.calibration_algorithms.probability_tree import ProbabilityTreeCalibrator

# Function to generate synthetic features, logits, and labels
def generate_synthetic_data(n_samples=1000, n_features=5, n_classes=3, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    X = np.random.randn(n_samples, n_features)  # Feature matrix
    logits = np.random.randn(n_samples, n_classes) * 3  # Simulated raw logits
    labels = np.random.choice(n_classes, n_samples)  # True labels
    
    return X, logits, labels

class TestProbabilityCalibrationTree(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Generate synthetic dataset for testing."""
        cls.X, cls.logits, cls.labels = generate_synthetic_data()
    
    def test_fit(self):
        """Test if the calibration tree fits without errors and updates metadata."""
        calibrator = ProbabilityTreeCalibrator(max_depth=3, min_samples_leaf=10)
        calibrator.fit(self.X, self.logits, self.labels)
        self.assertTrue(calibrator.fitted)
        
        # Validate metadata
        self.assertIn("dataset_info", calibrator.metadata)
        self.assertEqual(calibrator.metadata["dataset_info"].get("n_samples"), self.X.shape[0])
        self.assertEqual(calibrator.metadata["dataset_info"].get("n_features"), self.X.shape[1])
    
    def test_prediction_shape(self):
        """Ensure that the calibrated probabilities match the expected shape."""
        calibrator = ProbabilityTreeCalibrator(max_depth=3, min_samples_leaf=10)
        calibrator.fit(self.X, self.logits, self.labels)
        calibrated_probs = calibrator.predict(self.X, self.logits)
        self.assertEqual(calibrated_probs.shape, self.logits.shape)
        
    def test_probability_sum(self):
        """Check if the calibrated probabilities sum to approximately 1."""
        calibrator = ProbabilityTreeCalibrator(max_depth=3, min_samples_leaf=10)
        calibrator.fit(self.X, self.logits, self.labels)
        calibrated_probs = calibrator.predict(self.X, self.logits)
        sum_probs = np.sum(calibrated_probs, axis=1)
        self.assertTrue(np.allclose(sum_probs, np.ones(len(self.X)), atol=1e-4))
    
    def test_comparison_before_after_calibration(self):
        """Ensure calibration modifies the probability distribution."""
        softmax_probs = softmax(self.logits, axis=1)  # Uncalibrated probabilities
        calibrator = ProbabilityTreeCalibrator(max_depth=3, min_samples_leaf=10)
        calibrator.fit(self.X, self.logits, self.labels)
        calibrated_probs = calibrator.predict(self.X, self.logits)
        
        # Ensure the calibrated probabilities are different from uncalibrated ones
        self.assertFalse(np.allclose(calibrated_probs, softmax_probs))
    
    def test_runtime_error_if_not_fitted(self):
        """Ensure that predict raises an error if fit is not called first."""
        calibrator = ProbabilityTreeCalibrator(max_depth=3, min_samples_leaf=10)
        with self.assertRaises(RuntimeError):
            calibrator.predict(self.X, self.logits)
    
    def test_invalid_input_handling(self):
        """Ensure that invalid input shapes raise appropriate errors."""
        calibrator = ProbabilityTreeCalibrator(max_depth=3, min_samples_leaf=10)
        calibrator.fit(self.X, self.logits, self.labels)
        
        with self.assertRaises(ValueError):
            calibrator.predict(self.X[:, :2], self.logits)  # Invalid feature shape
        
        with self.assertRaises(ValueError):
            calibrator.predict(self.X, self.logits[:, :2])  # Invalid logits shape

if __name__ == "__main__":
    unittest.main()
