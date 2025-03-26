from .base import BaseCalibrationMetric
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfECE(BaseCalibrationMetric):
    """Class for computing the Confidence-based Expected Calibration Error (ConfECE).

    This metric computes the Expected Calibration Error (ECE) but only considers predictions
    with confidence above a specified threshold.

    Attributes:
        num_bins (int): The number of bins to use for calibration metrics.
        confidence_threshold (float): The confidence threshold for considering predictions.
        metric_name (str): The name of the metric.
    """

    def __init__(self, num_bins: int, confidence_threshold: float):
        """Initialize the ConfECE.

        Args:
            num_bins (int): The number of bins to use for calibration metrics.
            confidence_threshold (float): The confidence threshold for considering predictions.
        """
        super().__init__(num_bins)
        self.confidence_threshold = confidence_threshold
        self.metric_name = "ConfECE"

    def compute(self, predicted_probabilities, predicted_labels, true_labels) -> float:
        """Compute the Confidence-based Expected Calibration Error (ConfECE).

        Args:
            predicted_probabilities: The predicted probabilities.
            predicted_labels: The predicted labels.
            true_labels: The true labels.

        Returns:
            float: The computed ConfECE value.

        Raises:
            ValueError: If the lengths of predicted_probabilities and true_labels do not match.
        """
        predicted_probabilities = self._convert_to_numpy(predicted_probabilities)
        predicted_labels = self._convert_to_numpy(predicted_labels)
        true_labels = self._convert_to_numpy(true_labels)

        if not self.validate(predicted_probabilities, true_labels):
            raise ValueError("Input lengths mismatch")

        probs = predicted_probabilities
        preds = predicted_labels
        trues = true_labels

        if probs.ndim == 1:
            probs = np.vstack([1 - probs, probs]).T

        n_samples, _ = probs.shape
        confidences = probs[np.arange(n_samples), preds]
        accuracies = (preds == trues).astype(float)

        mask = confidences >= self.confidence_threshold
        confidences = confidences[mask]
        accuracies = accuracies[mask]

        if len(confidences) == 0:
            self.metric_value = 0.0
            return 0.0

        bins = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.clip(np.digitize(confidences, bins[:-1], right=False) - 1, 0, self.num_bins - 1)

        ece = 0.0
        total = len(confidences)
        for b in range(self.num_bins):
            mask_bin = bin_indices == b
            if np.sum(mask_bin) == 0:
                continue
            avg_conf = np.mean(confidences[mask_bin])
            avg_acc = np.mean(accuracies[mask_bin])
            ece += np.abs(avg_acc - avg_conf) * np.sum(mask_bin)

        self.metric_value = ece / total
        return self.metric_value

    def logger(self):
        """Log the metric name, value, and parameters."""
        log_data = {
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "parameters": {
                "num_bins": self.num_bins,
                "confidence_threshold": self.confidence_threshold
            }
        }
        logger.info(log_data)
        return log_data
