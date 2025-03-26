import numpy as np

from calibration_algorithms.calibration_base import CalibratorBase
from config.configuration_manager import ConfigurationManager
from utils.timer import time_operation
from config.enums.histogram_calibrator_types_enum import HistogramTypesEnum


config_manager = ConfigurationManager()

class HistogramCalibrator(CalibratorBase):
    """
    Implements Histogram for multi-class calibration.
    """
    def __init__(self, calibrator_type=config_manager.histogram_default_type, num_bins=config_manager.n_bin_cal, seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed)
        
        if not isinstance(calibrator_type, str):
            raise ValueError(f"Invalid calibrator type: {calibrator_type}. Must be a string.")

        try:
            self.calibrator_type = HistogramTypesEnum[calibrator_type.upper()]
        except KeyError:
            raise ValueError(f"Unknown calibrator type: {calibrator_type}. Available options: {list(HistogramTypesEnum.__members__.keys())}")

        self.num_bins = num_bins
        self.calibrator = None
        self.fitted = False
        self.timing = {}

    @time_operation
    def fit(self, predictions, ground_truth):
        """
        Trains the Histogram calibrator.

        Args:
            predictions (np.ndarray): Model logits of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        """
        self.set_random_seed(self.seed)  # Ensure full determinism

        predictions, ground_truth = self.validate_inputs(predictions, ground_truth)

        self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": predictions.shape[1],
        })
        self.metadata["params"].update({
            "Histogram Calibration Type": self.calibrator_type.name,
            "num_bins": self.num_bins
        })

        num_samples = len(ground_truth)  # Required for instantiation

        if self.calibrator_type == HistogramTypesEnum.HISTOGRAM:
            self.calibrator = self.calibrator_type.value(num_bins=self.num_bins, seed=self.seed)
            self.calibrator.fit(predictions, ground_truth)
        else:
            # Dynamically instantiate the correct calibrator model with both arguments
            self.calibrator = self.calibrator_type.value(num_calibration=num_samples, num_bins=self.num_bins)  
            self.calibrator.train_calibration(predictions, ground_truth)

        self.fitted = True

    @time_operation
    def predict(self, test_data):
        """
        Applies Histogram calibration to logits.

        Args:
            test_data (np.ndarray): Logits of shape (n_samples, n_classes).

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if not isinstance(test_data, np.ndarray):
            raise ValueError("Test data must be a NumPy array.")

        if test_data.ndim != 2:
            raise ValueError(f"Test data must be a 2D NumPy array, but got shape {test_data.shape}.")
        
        # Apply the correct calibration method based on HistogramTypesEnum
        if self.calibrator_type == HistogramTypesEnum.HISTOGRAM:
            calibrated_prob = self.calibrator.predict(test_data)
        else:
            calibrated_prob = self.calibrator.calibrate(test_data)

        return calibrated_prob
    
    def get_timing(self):
        return self.timing