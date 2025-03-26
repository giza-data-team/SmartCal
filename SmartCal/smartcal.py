import numpy as np
from enum import Enum
from metrics.ece import ECE
from metrics.conf_ece import ConfECE
from metrics.mce import MCE
from utils.functions import convert_one_hot_to_labels  # Import the utility function


class SmartCal:
    """
    Main class for handling model calibration workflow.
    It integrates multiple calibration metrics while allowing users to use them separately.
    """

    def __init__(self):
        """
        Initialize the GizaCal interface.
        """
        self.recommended_calibrators = None
        self.fitted_calibrator = None

    def recommend_calibrators(self, y_true: np.ndarray, predictions_prob: np.ndarray, n: int):
        """
        Recommend the top `n` calibration methods based on a given metric.

        :param y_true: A NumPy array containing the ground truth class labels (one-hot or single-label).
        :param predictions_prob: A NumPy array of predicted class probabilities for each sample.
        :param n: The number of calibration methods to recommend.
        :return: A list of recommended calibrators.
        """
        y_true = convert_one_hot_to_labels(y_true)  # Use utility function
        pass  # Will call ECE, ConfECE, and MCE for evaluation

    def best_fitted_calibrator(self, y_true: np.ndarray, predictions_prob: np.ndarray, metric: str):
        """
        Find the best calibration method using cross-validation.

        :param y_true: A NumPy array containing the ground truth class labels (one-hot or single-label).
        :param predictions_prob: A NumPy array of predicted class probabilities for each sample.
        :param metric: The performance metric used to determine the best calibrator.
        :return: The best fitted calibration method after evaluation.
        """
        y_true = convert_one_hot_to_labels(y_true)  # Use utility function
        pass  # Will use cross-validation across available calibrators

    def fit(self, y_true: np.ndarray, predictions_prob: np.ndarray, calibrator):
        """
        Fit a specific calibration method.

        :param y_true: A NumPy array containing the ground truth class labels (one-hot or single-label).
        :param predictions_prob: A NumPy array of predicted class probabilities for each sample.
        :param calibrator: An instance of a calibration method to be fitted.
        """
        y_true = convert_one_hot_to_labels(y_true)  # Use utility function
        self.fitted_calibrator = calibrator
        pass  # Will apply selected calibrator

    def calibrate(self, predictions_prob: np.ndarray) -> np.ndarray:
        """
        Apply the fitted calibration method.

        :param predictions_prob: A NumPy array of predicted class probabilities for each sample.
        :return: A NumPy array of calibrated probabilities.
        """
        if self.fitted_calibrator is None:
            raise RuntimeError("No calibrator has been fitted. Call `fit()` first.")
        pass  # Will apply calibration

    def evaluate_metric(self, y_true: np.ndarray, predictions_prob: np.ndarray, predictions_labels: np.ndarray,
                        metric_name: str, num_bins: int, confidence_threshold: float = None) -> float:
        """
        Compute a calibration metric independently.

        :param y_true: A NumPy array containing the ground truth class labels (one-hot or single-label).
        :param predictions_prob: A NumPy array of predicted class probabilities for each sample.
        :param predictions_labels: A NumPy array of predicted class labels.
        :param metric_name: The name of the calibration metric to compute ("ece", "conf_ece", "mce").
        :param num_bins: The number of bins to use when computing the metric.
        :param confidence_threshold: (Optional) The confidence threshold used for computing ConfECE.
        :return: The computed calibration metric value.
        """
        y_true = convert_one_hot_to_labels(y_true)  # Use utility function

        if metric_name.lower() == "ece":
            metric = ECE(num_bins=self.num_bins)
        elif metric_name.lower() == "conf_ece":
            metric = ConfECE(num_bins=self.num_bins,
                             confidence_threshold=confidence_threshold or 0.5)  # Default threshold
        elif metric_name.lower() == "mce":
            metric = MCE(num_bins=self.num_bins)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

        return metric.compute(predictions_prob, predictions_labels, y_true)
