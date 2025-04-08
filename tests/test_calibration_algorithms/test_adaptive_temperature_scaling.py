import unittest
import numpy as np
import torch
from Package.src.SmartCal.calibration_algorithms.adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator


class TestAdaptiveTemperatureScaling(unittest.TestCase):
    def setUp(self):
        """
        Initialize test data and the AdaptiveTemperatureScaling instance.
        """
        self.logits = np.array([
            [2.5, 1.0, -1.2],
            [1.2, 3.5, 0.5],
            [0.5, -0.5, 3.0]
        ])
        self.ground_truth = np.array([0, 1, 2])

    def test_initialization(self):
        """Test if the calibrator initializes with correct attributes."""
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            self.assertEqual(calibrator.mode, mode)
            self.assertFalse(calibrator.fitted)
            self.assertIsNone(calibrator.confidence_boundaries)
            self.assertIsNone(calibrator.entropy_boundaries)

    def test_compute_entropy(self):
        """Test the entropy computation method."""
        probs = torch.nn.functional.softmax(torch.tensor(self.logits, dtype=torch.float32), dim=1)
        calibrator = AdaptiveTemperatureScalingCalibrator(mode='hybrid')
        entropy = calibrator._compute_entropy(probs)
        self.assertEqual(entropy.shape[0], self.logits.shape[0])

    def test_fit(self):
        """Test if the fit method properly calibrates the model."""
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            calibrator.fit(self.logits, self.ground_truth)
            self.assertTrue(calibrator.fitted)
            if mode in ['hybrid', 'linear']:
                self.assertIsNotNone(calibrator.confidence_boundaries)
            elif mode in ['hybrid','entropy']:
                self.assertIsNotNone(calibrator.entropy_boundaries)

    def test_predict(self):
        """Test the prediction method after fitting."""
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            calibrator.fit(self.logits, self.ground_truth)
            calibrated_probs = calibrator.predict(self.logits)
            self.assertEqual(calibrated_probs.shape, self.logits.shape)
            self.assertTrue(np.all(calibrated_probs >= 0) and np.all(calibrated_probs <= 1))
            self.assertAlmostEqual(np.sum(calibrated_probs, axis=1).all(), 1.0, places=5)

    def test_unfitted_predict(self):
        """Test that predict raises an error if called before fitting."""
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            with self.assertRaises(RuntimeError):
                calibrator.predict(self.logits)

    def test_calibration_effect(self):
        """Test that calibration improves or maintains log loss."""
        from sklearn.metrics import log_loss
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            initial_probs = torch.nn.functional.softmax(torch.tensor(self.logits), dim=1).numpy()
            initial_log_loss = log_loss(self.ground_truth, initial_probs)
            calibrator.fit(self.logits, self.ground_truth)
            calibrated_probs = calibrator.predict(self.logits)
            calibrated_log_loss = log_loss(self.ground_truth, calibrated_probs)
            self.assertLessEqual(calibrated_log_loss, initial_log_loss)

    def test_extreme_logits(self):
        """Test that the model handles extreme logits without numerical instability."""
        extreme_logits = np.array([
            [1000, -1000, 0],
            [-1000, 1000, 0],
            [0, 0, 1000]
        ])
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            calibrator.fit(extreme_logits, self.ground_truth)
            calibrated_probs = calibrator.predict(extreme_logits)
            self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5))

    def test_large_dataset(self):
        """Test performance and correctness with a large dataset."""
        large_logits = np.random.randn(10000, 3)
        large_labels = np.random.randint(0, 3, size=10000)
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            calibrator.fit(large_logits, large_labels)
            calibrated_probs = calibrator.predict(large_logits)
            self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5))

    def test_invalid_predict_inputs(self):
        """Test that predict raises errors for invalid inputs."""
        for mode in ['hybrid', 'linear', 'entropy']:
            calibrator = AdaptiveTemperatureScalingCalibrator(mode=mode)
            calibrator.fit(self.logits, self.ground_truth)
            with self.assertRaises(ValueError):
                calibrator.predict([[0.1, 0.9]])  # Non-NumPy array
            with self.assertRaises(ValueError):
                calibrator.predict(np.array([0.1, 0.9]))  # 1D array instead of 2D


if __name__ == '__main__':
    unittest.main()