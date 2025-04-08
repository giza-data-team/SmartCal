import unittest
import numpy as np
import torch
from Package.src.SmartCal.calibration_algorithms.vector_scaling import VectorScalingCalibrator


class TestVectorScaling(unittest.TestCase):
    def setUp(self):
        """
        Initialize test data and the VectorScaling instance.
        """
        self.logits = np.array([
            [2.5, 1.0, -1.2],
            [1.2, 3.5, 0.5],
            [0.5, -0.5, 3.0]
        ])
        self.ground_truth = np.array([0, 1, 2])
        self.vs = VectorScalingCalibrator(lr=0.01, max_iter=100)

    def test_fit(self):
        """
        Test that fit correctly optimizes the scaling parameters.
        """
        self.vs.fit(self.logits, self.ground_truth)
        self.assertTrue(self.vs.fitted, "Model should be marked as fitted.")
        self.assertIsNotNone(self.vs.W, "Optimized weight matrix should be stored.")
        self.assertIsNotNone(self.vs.b, "Optimized bias vector should be stored.")

    def test_predict(self):
        """
        Test that predict outputs calibrated probabilities.
        """
        self.vs.fit(self.logits, self.ground_truth)
        calibrated_probs = self.vs.predict(self.logits)
        self.assertIsInstance(calibrated_probs, np.ndarray, "Calibrated probabilities should be a NumPy array.")
        self.assertEqual(calibrated_probs.shape, self.logits.shape, "Output shape should match input logits.")
        self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5),
                        "Calibrated probabilities should sum to 1 for each sample.")

    def test_not_fitted_error(self):
        """
        Test that predict raises an error if fit is not called first.
        """
        with self.assertRaises(RuntimeError):
            self.vs.predict(self.logits)

    def test_calibration_effect(self):
        """
        Test that calibration improves or maintains log loss.
        """
        from sklearn.metrics import log_loss

        initial_probs = torch.nn.functional.softmax(torch.tensor(self.logits), dim=1).numpy()
        initial_log_loss = log_loss(self.ground_truth, initial_probs)

        self.vs.fit(self.logits, self.ground_truth)
        calibrated_probs = self.vs.predict(self.logits)
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
        self.vs.fit(extreme_logits, self.ground_truth)
        calibrated_probs = self.vs.predict(extreme_logits)

        self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5),
                        "Calibrated probabilities should sum to 1 for extreme logits.")

    def test_large_dataset(self):
        """
        Test performance and correctness with a large dataset.
        """
        large_logits = np.random.randn(10000, 3)  # Simulated large dataset
        large_labels = np.random.randint(0, 3, size=10000)
        self.vs.fit(large_logits, large_labels)
        calibrated_probs = self.vs.predict(large_logits)

        self.assertTrue(np.allclose(calibrated_probs.sum(axis=1), 1, atol=1e-5),
                        "Calibrated probabilities should sum to 1 for large datasets.")

    def test_multiple_fit_calls(self):
        """
        Test that multiple calls to fit work as expected.
        """
        vs = VectorScalingCalibrator(lr=0.01, max_iter=100)

        # First fit call
        vs.fit(self.logits, self.ground_truth)
        first_W, first_b = vs.W, vs.b

        # Slightly modify the logits for the second fit call
        modified_logits = self.logits + np.random.uniform(-0.01, 0.01, self.logits.shape)

        vs.fit(modified_logits, self.ground_truth)
        second_W, second_b = vs.W, vs.b

        self.assertFalse(np.array_equal(first_W, second_W) or np.array_equal(first_b, second_b),
                         "Weights and biases should be re-optimized on subsequent fit calls with different data.")

    def test_invalid_predict_inputs(self):
        """
        Test that predict raises errors for invalid inputs.
        """
        self.vs.fit(self.logits, self.ground_truth)
        with self.assertRaises(ValueError):
            self.vs.predict([[0.1, 0.9]])  # Non-NumPy array
        with self.assertRaises(ValueError):
            self.vs.predict(np.array([0.1, 0.9]))  # 1D array instead of 2D

    def test_deterministic_behavior(self):
        """
        Test that the model is deterministic with the same random seed.
        """
        torch.manual_seed(42)
        self.vs.fit(self.logits, self.ground_truth)
        first_W, first_b = self.vs.W, self.vs.b

        torch.manual_seed(42)
        self.vs.fit(self.logits, self.ground_truth)
        second_W, second_b = self.vs.W, self.vs.b

        self.assertTrue(
            np.allclose(first_W.detach().numpy(), second_W.detach().numpy(), atol=1e-6) and
            np.allclose(first_b.detach().numpy(), second_b.detach().numpy(), atol=1e-6),
            "Model should produce deterministic results with the same seed."
        )

if __name__ == '__main__':
    unittest.main()
