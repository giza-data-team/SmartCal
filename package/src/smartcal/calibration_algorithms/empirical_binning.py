import numpy as np
from typing import Optional

from smartcal.calibration_algorithms.calibration_base import CalibratorBase
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.utils.timer import time_operation


config_manager = ConfigurationManager()

class EmpiricalBinningCalibrator(CalibratorBase):
    def __init__(self, n_bins: int = config_manager.n_bin_cal):
        super().__init__()
        self.validate_init_params(n_bins)

        self.n_bins = n_bins
        self.bin_boundaries = None
        self.bin_probabilities = None
        self.timing = {}

        self.logger.info(f"Initialized EmpiricalBinningCalibrator with {n_bins} bins")

        # Update metadata with specific parameters
        self.metadata.update({
            'calibration_type': 'empirical_binning'
        })
        self.metadata["params"].update({
            'n_bins': n_bins
        })

    def validate_init_params(self, n_bins: int) -> None:
        if not isinstance(n_bins, int):
            raise TypeError("n_bins must be an integer")
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")

    def validate_input(self, predictions, ground_truth=None):
        """
        Extends the base class validation with multiclass classification checks.
        """
        # First use the base class validation
        if ground_truth is not None:
            predictions, ground_truth = super().validate_inputs(predictions, ground_truth)

        # Ensure predictions are properly shaped
        if predictions.ndim != 2:
            raise ValueError("Predictions should be a 2D array (n_samples, n_classes)")

        n_classes = predictions.shape[1]
        if np.any((predictions < 0) | (predictions > 1)):
            raise ValueError("Predictions must be in range [0, 1]")

        if ground_truth is not None:
            unique_labels = np.unique(ground_truth)
            if not np.all(np.isin(unique_labels, np.arange(n_classes))):
                raise ValueError(f"Ground truth labels must be integers from 0 to {n_classes - 1}")

        return predictions, ground_truth


    def compute_bin_statistics(self, predictions, ground_truth: Optional[np.ndarray] = None):
        try:
            """
            Compute bin statistics including boundaries and mean probabilities.
            """

            n_classes = predictions.shape[1]
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            bin_probabilities = np.zeros((self.n_bins, n_classes))
            bin_counts = np.zeros(self.n_bins)  # Track number of samples in each bin

            for class_idx in range(n_classes):
                class_preds = predictions[:, class_idx]
                bin_indices = np.digitize(class_preds, bin_boundaries, right=False) - 1
                bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)  # Fix indexing issue

                for i in range(self.n_bins):
                    bin_mask = bin_indices == i
                    bin_counts[i] += np.sum(bin_mask)

                    if np.any(bin_mask):
                        # Use empirical probability (fraction of true positives)
                        bin_probabilities[i, class_idx] = np.mean(ground_truth[bin_mask] == class_idx)
                    else:
                        # Mark empty bins for interpolation
                        bin_probabilities[i, class_idx] = np.nan

            # Handle empty bins through interpolation
            for class_idx in range(n_classes):
                empty_bins = np.isnan(bin_probabilities[:, class_idx])
                valid_bins = ~empty_bins

                if np.all(empty_bins):
                    bin_probabilities[:, class_idx] = 0  # Default fallback

                elif np.any(empty_bins) and np.any(valid_bins):
                    # Use linear interpolation
                    x = np.where(valid_bins)[0]
                    y = bin_probabilities[valid_bins, class_idx]
                    empty_x = np.where(empty_bins)[0]

                    bin_probabilities[empty_bins, class_idx] = np.interp(empty_x, x, y)

                # Enforce monotonicity
                bin_probabilities[:, class_idx] = np.maximum.accumulate(bin_probabilities[:, class_idx])

            # Log statistics
            self.logger.info(f"Bin sample counts: {bin_counts}")
            self.logger.info(f"Empty bins: {np.sum(bin_counts == 0)}")

            return bin_boundaries, bin_probabilities

        except Exception as e:
            self.logger.error(f"Error in compute_bin_statistics: {str(e)}")
            raise

    @time_operation
    def fit(self, predictions: np.ndarray, ground_truth):
        """
        Fit the calibrator using uncalibrated predictions and true labels.
        """
        try:
            predictions, ground_truth = self.validate_input(predictions, ground_truth)
            n_classes = predictions.shape[1]

            # Log dataset details
            self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": predictions.shape[1],
            })

            self.logger.info(f"Starting calibration fitting with parameters: {self.metadata}")

            self.bin_boundaries, self.bin_probabilities = self.compute_bin_statistics(predictions, ground_truth)
            self.fitted = True

            # Log calibration parameters
            self.metadata['calibration_params'] = {
                'bin_boundaries': self.bin_boundaries.tolist(),
                'bin_probabilities': self.bin_probabilities.tolist()
            }

            self.logger.info(f"Calibration parameters: {self.metadata['calibration_params']}")
            self.logger.info("Successfully fitted EmpiricalBinningCalibrator")
            return self

        except Exception as e:
            self.logger.error(f"Error during fitting: {str(e)}")
            raise

    @time_operation
    def predict(self, predictions):
        """
        Calibrate predictions using the fitted parameters while preserving monotonicity.
        """   
        if not self.fitted:
            raise ValueError("Calibrator must be fitted before making predictions")

        try:
            predictions, _ = self.validate_input(predictions)

            calibrated_probs = np.zeros_like(predictions)

            for class_idx in range(predictions.shape[1]):
                class_preds = predictions[:, class_idx]
                class_bin_indices = np.digitize(class_preds, self.bin_boundaries, right=False) - 1
                class_bin_indices = np.clip(class_bin_indices, 0, self.n_bins - 1)

                calibrated_probs[:, class_idx] = self.bin_probabilities[class_bin_indices, class_idx]

            # Normalize so that probabilities sum to 1
            row_sums = np.sum(calibrated_probs, axis=1, keepdims=True)
            valid_rows = row_sums > 0
            calibrated_probs[valid_rows[:, 0], :] /= row_sums[valid_rows[:, 0], :]

            # Handle invalid (zero) rows: fallback to uniform
            invalid_rows = ~valid_rows[:, 0]
            if np.any(invalid_rows):
                calibrated_probs[invalid_rows, :] = 1.0 / predictions.shape[1]

            if not np.all((calibrated_probs >= 0) & (calibrated_probs <= 1)):
                raise ValueError("Calibrated probabilities out of range")

            if not np.allclose(np.sum(calibrated_probs, axis=1), 1, atol=1e-4):
                raise ValueError("Probabilities do not sum to 1 after normalization")

            return calibrated_probs

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def get_timing(self):
        """Return the timing information for fit and predict operations."""
        return self.timing
