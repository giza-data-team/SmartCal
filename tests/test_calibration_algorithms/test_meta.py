import unittest
import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import entropy

from calibration_algorithms.meta import MetaCalibrator


# Function to generate synthetic logits and labels
def generate_synthetic_data(n_samples=1000, n_classes=3, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    logits = np.random.randn(n_samples, n_classes) * 3  # Simulating raw logits
    labels = np.random.choice(n_classes, n_samples)  # True labels

    return logits, labels

class TestMetaCalibrator(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Generate synthetic dataset for testing."""
        cls.logits, cls.labels = generate_synthetic_data()
    
    def test_miscoverage_mode(self):
        """Test MetaCalibrator with Miscoverage Constraint (alpha)."""
        calibrator = MetaCalibrator(calibrator_type='ALPHA',alpha=0.1, acc=0.08)
        calibrator.fit(self.logits, self.labels)
        calibrated_probs = calibrator.predict(self.logits)
        
        # Check metadata
        self.assertIn("calibration_type", calibrator.metadata)
        self.assertIn("params", calibrator.metadata)
        self.assertIn("dataset_info", calibrator.metadata)
        self.assertEqual(calibrator.metadata["params"].get("alpha"), 0.1)
        self.assertEqual(calibrator.metadata["params"].get("acc"), None)
        
        # Check output shape
        self.assertEqual(np.array(calibrated_probs).shape, np.array(self.logits).shape)
        
        # Ensure probabilities sum to ~1
        self.assertTrue(np.allclose(np.sum(calibrated_probs, axis=1), np.ones(len(self.logits)), atol=1e-4))
    
    def test_coverage_accuracy_mode(self):
        """Test MetaCalibrator with Coverage Accuracy Constraint (acc)."""
        calibrator = MetaCalibrator(calibrator_type='ACC',alpha=0.1, acc=0.85)
        calibrator.fit(self.logits, self.labels)
        calibrated_probs = calibrator.predict(self.logits)
        
        # Check metadata
        self.assertIn("calibration_type", calibrator.metadata)
        self.assertIn("params", calibrator.metadata)
        self.assertIn("dataset_info", calibrator.metadata)
        self.assertEqual(calibrator.metadata["params"].get("alpha"), None)
        self.assertEqual(calibrator.metadata["params"].get("acc"), 0.85)
        
        # Check output shape
        self.assertEqual(np.array(calibrated_probs).shape, np.array(self.logits).shape)
        
        # Ensure probabilities sum to ~1
        self.assertTrue(np.allclose(np.sum(calibrated_probs, axis=1), np.ones(len(self.logits)), atol=1e-4))
    
    def test_comparison_before_after_calibration(self):
        """Compare original vs. calibrated predictions."""
        softmax_probs = softmax(self.logits, axis=1)  # Uncalibrated probabilities
        calibrator = MetaCalibrator(alpha=0.1)
        calibrator.fit(self.logits, self.labels)
        calibrated_probs = calibrator.predict(self.logits)
        
        # Ensure shape consistency
        self.assertEqual(np.array(calibrated_probs).shape, np.array(softmax_probs).shape)
        
        # Ensure probabilities have been adjusted
        self.assertFalse(np.allclose(calibrated_probs, softmax_probs))
    
    def test_calibration_performance(self):
        """Evaluate calibration performance using accuracy and entropy."""
        def evaluate_calibration(predictions, labels):
            correct = np.argmax(predictions, axis=1) == labels
            accuracy = np.mean(correct)
            entropy_scores = entropy(predictions, axis=1)
            avg_entropy = np.mean(entropy_scores)
            return accuracy, avg_entropy
        
        softmax_probs = softmax(self.logits, axis=1)  # Original uncalibrated probabilities
        calibrator = MetaCalibrator(alpha=0.1)
        calibrator.fit(self.logits, self.labels)
        calibrated_probs = calibrator.predict(self.logits)
        original_acc, original_entropy = evaluate_calibration(softmax_probs, self.labels)
        calibrated_acc, calibrated_entropy = evaluate_calibration(calibrated_probs, self.labels)
        
        # Check that accuracy and entropy are reasonable values
        self.assertGreaterEqual(original_acc, 0)
        self.assertLessEqual(original_acc, 1)
        self.assertGreaterEqual(calibrated_acc, 0)
        self.assertLessEqual(calibrated_acc, 1)
        
        # Entropy should generally increase after calibration (correction of overconfidence)
        self.assertGreaterEqual(calibrated_entropy, original_entropy)

if __name__ == "__main__":
    unittest.main()
