import unittest
import numpy as np
from unittest.mock import patch
from Package.src.SmartCal.SmartCal.SmartCal import SmartCal

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax for a batch of logits."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def generate_sample_data(n_samples: int = 50, n_classes: int = 3, seed: int = 42):
    """Generate synthetic logits and true labels for demo purposes."""
    np.random.seed(seed)
    logits = np.random.randn(n_samples, n_classes) * 2
    probabilities = softmax(logits)
    y_true = np.random.choice(n_classes, size=n_samples)
    return y_true, probabilities


class TestSmartCal(unittest.TestCase):

    def setUp(self):
        self.smartcal = SmartCal()
        self.y_true, self.predictions_prob = generate_sample_data()

    @patch("meta_model.meta_model.MetaModel")
    @patch("meta_features_extraction.meta_features_extraction.MetaFeaturesExtractor")
    def test_recommend_calibrators(self, MockExtractor, MockModel):
        # Mock meta-feature extractor and meta-model
        mock_extractor = MockExtractor.return_value
        mock_extractor.process_features.return_value = "mocked_meta_features"

        mock_model = MockModel.return_value
        mock_model.predict_best_model.return_value = [("PLATT", 0.6), ("HISTOGRM", 0.4)]

        recommendations = self.smartcal.recommend_calibrators(
            self.y_true, self.predictions_prob, n=2, metric="ECE"
        )

        self.assertEqual(len(recommendations), 2)
        self.assertAlmostEqual(sum(score for _, score in recommendations), 1.0)

    @patch("autocal_end_to_end.calibrators_bayesian_optimization.CalibrationOptimizer")
    @patch("meta_model.meta_model.MetaModel")
    @patch("meta_features_extraction.meta_features_extraction.MetaFeaturesExtractor")
    def test_best_fitted_calibrator(self, MockExtractor, MockBModel, MockOptimizer):
        # Pretend we've already called recommend_calibrators
        self.smartcal.recommended_calibrators = [("PLATT", 0.7), ("HISTOGRM", 0.3)]

        mock_optimizer = MockOptimizer.return_value
        mock_optimizer.allocate_iterations.return_value = [7, 3]

        mock_optimizer.optimize_calibrator.return_value = {
            "best_params": {},
            "full_metrics": True
        }

        # Dummy calibrator
        class DummyCalibrator:
            def fit(self, X, y): pass
            def predict(self, X): return X

        with patch("config.enums.calibration_algorithms_enum.CalibrationAlgorithmTypesEnum") as MockEnum:
            MockEnum.__getitem__.return_value = DummyCalibrator
            best_cal = self.smartcal.best_fitted_calibrator(self.y_true, self.predictions_prob, n_iter=10, metric='ECE')

        self.assertIsNotNone(best_cal)


if __name__ == "__main__":
    unittest.main()
