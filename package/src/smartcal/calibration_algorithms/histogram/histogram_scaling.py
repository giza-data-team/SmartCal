import numpy as np
import bisect
from typing import List, TypeVar

from smartcal.calibration_algorithms.calibration_base import CalibratorBase
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.utils.timer import time_operation


config_manager = ConfigurationManager()

class HistogramScalingCalibrator(CalibratorBase):
    """
    Implements histogram binning for multi-class probability calibration.
    """
    def __init__(self, num_bins=config_manager.n_bin_cal, seed=CalibratorBase.DEFAULT_SEED):
        """
        Initialize the Histogram Binning calibrator.
        :param num_bins: Number of bins to use for histogram binning.
        """
        super().__init__(seed)
        self.num_bins = num_bins
        self.histogram_models = None
        self.bins = None
        self.n_classes = None
        self.metadata.update({"calibration_type": "histogram_binning"})  # Store calibration type
        self.metadata["params"].update({"num_bins": self.num_bins})  
        self.timing = {}

    @time_operation
    def fit(self, logits, labels):
        """
        Train the Histogram Binning calibrator.
        :param logits: (n_samples, n_classes) array of model logits or confidence scores.
        :param labels: (n_samples,) array of true labels.
        """
        logits, labels = super().validate_inputs(logits, labels)  # Ensure input consistency
        logits = np.array(logits)
        labels = np.array(labels)
        n_samples, self.n_classes = logits.shape

        self.histogram_models = []
        self.bins = []

        # Log dataset info
        self.metadata["dataset_info"].update({"n_samples": n_samples, "n_classes": self.n_classes})

        # One-vs-All Histogram Binning
        for class_idx in range(self.n_classes):
            # Get class-wise probabilities
            class_probs = logits[:, class_idx]
            class_labels = (labels == class_idx).astype(float)  # Convert labels to binary (One-vs-All)

            # Compute bin edges
            try:
                bins = get_equal_bins(class_probs, num_bins=self.num_bins)
            except ValueError as e:
                self.logger.error(f"Bin computation failed for class {class_idx}: {e}")
                raise RuntimeError(f"Failed to compute bins for class {class_idx}. Reduce `num_bins`.")

            # Train histogram calibrator
            try:
                histogram_model = get_histogram_calibrator(class_probs, class_labels, bins)
            except ValueError as e:
                self.logger.error(f"Histogram calibration failed for class {class_idx}: {e}")
                raise RuntimeError(f"Histogram calibration failed for class {class_idx}.")

            self.bins.append(bins)
            self.histogram_models.append(histogram_model)

        self.fitted = True
        self.logger.info(f"HistogramCalibrator fitted with {self.num_bins} bins per class.")

    @time_operation
    def predict(self, logits):
        """
        Apply Histogram Binning calibration to logits.
        :param logits: (n_samples, n_classes) array of logits or confidence scores.
        :return: (n_samples, n_classes) array of calibrated probabilities.
        """
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted. Call `fit()` first.")

        logits = np.array(logits)

        # Ensure input shape matches the number of classes
        if logits.shape[1] != self.n_classes:
            raise ValueError(f"Mismatch in number of classes: expected {self.n_classes}, got {logits.shape[1]}.")

        n_samples = logits.shape[0]
        calibrated_probs = np.zeros((n_samples, self.n_classes))

        # Apply histogram binning for each class
        for class_idx in range(self.n_classes):
            class_probs = logits[:, class_idx]
            calibrated_probs[:, class_idx] = self.histogram_models[class_idx](class_probs)

        # Normalize to ensure probabilities sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        calibrated_probs /= np.clip(row_sums, 1e-15, None)  # Avoid division by zero

        return calibrated_probs
    
    def get_timing(self):
        return self.timing

# Define data types
T = TypeVar('T')
eps = 1e-6

def split(sequence: List[T], parts: int) -> List[List[T]]:
    """
    Splits a sequence into approximately equal-sized parts.
    """
    assert parts <= len(sequence)
    array_splits = np.array_split(sequence, parts)
    return [list(l) for l in array_splits]

def get_bin(pred_prob: float, bins: List[float]) -> int:
    """
    Get the index of the bin that pred_prob belongs in.
    """
    assert 0.0 <= pred_prob <= 1.0
    assert bins[-1] == 1.0
    return bisect.bisect_left(bins, pred_prob)

def get_equal_bins(probs: List[float], num_bins: int = 10) -> List[float]:
    """
    Compute bin edges that contain approximately an equal number of samples.
    """
    sorted_probs = sorted(probs)
    if num_bins > len(sorted_probs):
        num_bins = len(sorted_probs)

    binned_data = split(sorted_probs, num_bins)
    bins = []
    for i in range(len(binned_data) - 1):
        last_prob = binned_data[i][-1]
        next_first_prob = binned_data[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)
    bins.append(1.0)
    return sorted(list(set(bins)))

def get_histogram_calibrator(model_probs, values, bins):
    """
    Creates a histogram-based calibration function.
    """
    binned_values = [[] for _ in range(len(bins))]

    for prob, value in zip(model_probs, values):
        bin_idx = get_bin(prob, bins)
        binned_values[bin_idx].append(float(value))

    def safe_mean(values, bin_idx):
        if len(values) == 0:
            return float(bins[bin_idx]) / 2.0 if bin_idx == 0 else float(bins[bin_idx] + bins[bin_idx - 1]) / 2.0
        return np.mean(values)

    bin_means = np.array([safe_mean(values, idx) for idx, values in enumerate(binned_values)])

    def calibrator(probs):
        indices = np.searchsorted(bins, probs)
        return bin_means[indices]

    return calibrator
