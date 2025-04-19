import numpy as np
import logging

from smartcal.metrics.ece import ECE


logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

class MCE(ECE):
    """Class for computing the Maximum Calibration Error (MCE).

    This metric measures the maximum difference between the predicted probabilities and the actual accuracy.

    Attributes:
        num_bins (int): The number of bins to use for calibration metrics.
        metric_name (str): The name of the metric.
    """

    def __init__(self, num_bins: int):
        """Initialize the MCE.

        Args:
            num_bins (int): The number of bins to use for calibration metrics.
        """
        super().__init__(num_bins)
        self.metric_name = "MCE"

    def compute(self, predicted_probabilities, predicted_labels, true_labels) -> float:
        """Compute the Maximum Calibration Error (MCE).

        Args:
            predicted_probabilities: The predicted probabilities.
            predicted_labels: The predicted labels.
            true_labels: The true labels.

        Returns:
            float: The computed MCE value.

        Raises:
            ValueError: If the lengths of predicted_probabilities and true_labels do not match.
        """
        predicted_probabilities = self._convert_to_numpy(predicted_probabilities)
        predicted_labels = self._convert_to_numpy(predicted_labels)
        true_labels = self._convert_to_numpy(true_labels)

        super().compute(predicted_probabilities, predicted_labels, true_labels)
        confidences = predicted_probabilities
        if confidences.ndim == 1:
            confidences = np.vstack([1 - confidences, confidences]).T

        preds = predicted_labels
        trues = true_labels

        n_samples = confidences.shape[0]
        confs = confidences[np.arange(n_samples), preds]
        accs = (preds == trues).astype(float)

        bins = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.clip(np.digitize(confs, bins[:-1], right=False) - 1, 0, self.num_bins - 1)

        mce = 0.0
        for b in range(self.num_bins):
            mask = bin_indices == b
            if np.sum(mask) == 0:
                continue
            avg_conf = np.mean(confs[mask])
            avg_acc = np.mean(accs[mask])
            mce = max(mce, np.abs(avg_acc - avg_conf))

        self.metric_value = mce
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
