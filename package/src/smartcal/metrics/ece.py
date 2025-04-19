import numpy as np
import logging

from smartcal.metrics.base import BaseCalibrationMetric


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class ECE(BaseCalibrationMetric):
    """Class for computing the Expected Calibration Error (ECE).

    This metric measures the difference between the predicted probabilities and the actual accuracy.

    Attributes:
        num_bins (int): The number of bins to use for calibration metrics.
        metric_name (str): The name of the metric.
    """

    def __init__(self, num_bins: int):
        """Initialize the ECE.

        Args:
            num_bins (int): The number of bins to use for calibration metrics.
        """
        super().__init__(num_bins)
        self.metric_name = "ECE"

    def compute(self, predicted_probabilities, predicted_labels, true_labels) -> float:
        """Compute the Expected Calibration Error (ECE).

        Args:
            predicted_probabilities: The predicted probabilities.
            predicted_labels: The predicted labels.
            true_labels: The true labels.

        Returns:
            float: The computed ECE value.

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

        n_samples, n_classes = probs.shape
        confidences = probs[np.arange(n_samples), preds]
        accuracies = (preds == trues).astype(float)

        bins = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.clip(np.digitize(confidences, bins[:-1], right=False) - 1, 0, self.num_bins - 1)

        ece = 0.0
        for b in range(self.num_bins):
            mask = bin_indices == b
            if np.sum(mask) == 0:
                continue
            avg_conf = np.mean(confidences[mask])
            avg_acc = np.mean(accuracies[mask])
            ece += np.abs(avg_acc - avg_conf) * np.sum(mask)

        self.metric_value = ece / n_samples
        return self.metric_value

    def logger(self):
        """Log the metric name, value, and parameters."""
        log_data = {
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "parameters": {
                "num_bins": self.num_bins
            }
        }
        logger.info(log_data)
        return log_data
