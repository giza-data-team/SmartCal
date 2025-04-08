import unittest
import numpy as np
import torch
from Package.src.SmartCal.calibration_algorithms.temperature_scaling import TemperatureScalingCalibrator


class TestTemperatureScaling(unittest.TestCase):
    def setUp(self):
        """
        Initialize test data and the TemperatureScaling instance.
        """
        self.logits = np.array([
            [2.5, 1.0, -1.2],
            [1.2, 3.5, 0.5],
            [0.5, -0.5, 3.0]
        ])
        self.ground_truth = np.array([0, 1, 2])
        self.ts = TemperatureScalingCalibrator(lr_tempscaling=0.01, max_iter_tempscaling=100)

    def test_fit(self):
        """
        Test that fit correctly optimizes the temperature parameter.
        """
        self.ts.fit(self.logits, self.ground_truth)
        self.assertTrue(self.ts.fitted, "Model should be marked as fitted.")
        self.assertIsNotNone(self.ts.optimized_temperature, "Optimized temperature should be stored.")
        self.assertGreater(self.ts.optimized_temperature, 0, "Optimized temperature must be positive.")

    def test_predict(self):
        """
        Test that predict outputs calibrated probabilities.
        """
        self.ts.fit(self.logits, self.ground_truth)
        calibrated_probs = self.ts.predict(self.logits)
        self.assertIsInstance(calibrated_probs, np.ndarray, "Calibrated probabilities should be a NumPy array.")
        self.assertEqual(calibrated_probs.shape, self.logits.shape, "Output shape should match input logits.")
        self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5),
                        "Calibrated probabilities should sum to 1 for each sample.")

    def test_not_fitted_error(self):
        """
        Test that predict raises an error if fit is not called first.
        """
        with self.assertRaises(RuntimeError):
            self.ts.predict(self.logits)

    def test_calibration_effect(self):
        """
        Test that calibration improves or maintains log loss.
        """
        from sklearn.metrics import log_loss

        initial_probs = torch.nn.functional.softmax(torch.tensor(self.logits), dim=1).numpy()
        initial_log_loss = log_loss(self.ground_truth, initial_probs)

        self.ts.fit(self.logits, self.ground_truth)
        calibrated_probs = self.ts.predict(self.logits)
        calibrated_log_loss = log_loss(self.ground_truth, calibrated_probs)

        self.assertLessEqual(calibrated_log_loss, initial_log_loss,
                             "Calibration should not increase log loss.")
        
    def test_extreme_logits(self):
        """
        Test that the model handles extreme logits without numerical instability.
        """
        extreme_logits = np.array([
            [1000, -1000, 0],
            [-1000, 1000, 0],
            [0, 0, 1000]
        ])
        self.ts.fit(extreme_logits, self.ground_truth)
        calibrated_probs = self.ts.predict(extreme_logits)

        self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5),
                        "Calibrated probabilities should sum to 1 for extreme logits.")
        
    def test_large_dataset(self):
        """
        Test performance and correctness with a large dataset.
        """
        large_logits = np.random.randn(10000, 3)  # Simulated large dataset
        large_labels = np.random.randint(0, 3, size=10000)
        self.ts.fit(large_logits, large_labels)
        calibrated_probs = self.ts.predict(large_logits)

        self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5),
                        "Calibrated probabilities should sum to 1 for large datasets.")

    def test_multiple_fit_calls(self):
        """
        Test that multiple calls to fit work as expected.
        """
        ts = TemperatureScalingCalibrator(lr_tempscaling=0.01, max_iter_tempscaling=100, seed=42)

        # First fit call
        ts.fit(self.logits, self.ground_truth)
        first_temperature = ts.optimized_temperature

        # Slightly modify the logits for the second fit call
        modified_logits = self.logits + np.random.uniform(-0.01, 0.01, self.logits.shape)

        ts.fit(modified_logits, self.ground_truth)
        second_temperature = ts.optimized_temperature

        self.assertNotEqual(first_temperature, second_temperature,
                            "Temperature should be re-optimized on subsequent fit calls with different data.")
  
    def test_invalid_predict_inputs(self):
        """
        Test that predict raises errors for invalid inputs.
        """
        self.ts.fit(self.logits, self.ground_truth)
        with self.assertRaises(ValueError):
            self.ts.predict([[0.1, 0.9]])  # Non-NumPy array
        with self.assertRaises(ValueError):
            self.ts.predict(np.array([0.1, 0.9]))  # 1D array instead of 2D

    def test_temperature_clamping(self):
        """
        Test that the temperature parameter is clamped to avoid division by zero.
        """
        self.ts.fit(self.logits, self.ground_truth)
        self.assertGreaterEqual(self.ts.optimized_temperature, 1e-6,
                                "Temperature should be clamped to avoid values below 1e-6.")
        
    def test_deterministic_behavior(self):
        """
        Test that the model is deterministic with the same random seed.
        """
        torch.manual_seed(42)
        self.ts.fit(self.logits, self.ground_truth)
        first_temperature = self.ts.optimized_temperature

        torch.manual_seed(42)
        self.ts.fit(self.logits, self.ground_truth)
        second_temperature = self.ts.optimized_temperature

        self.assertAlmostEqual(first_temperature, second_temperature,
                            places=6, msg="Model should produce deterministic results with the same seed.")

if __name__ == '__main__':
    unittest.main()
