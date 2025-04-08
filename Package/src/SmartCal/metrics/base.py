import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseCalibrationMetric(ABC):
    """Base class for calibration metrics.

    This class provides a framework for implementing calibration metrics.
    It includes methods for validating input data and converting data to numpy arrays.

    Attributes:
        num_bins (int): The number of bins to use for calibration metrics.
        metric_name (str): The name of the metric.
    """

    def __init__(self, num_bins: int):
        """Initialize the BaseCalibrationMetric.

        Args:
            num_bins (int): The number of bins to use for calibration metrics.

        Raises:
            ValueError: If num_bins is less than 1.
        """
        if num_bins < 1:
            raise ValueError("num_bins must be at least 1")
        self.num_bins = num_bins
        self.metric_name = None
        self.metric_value = None

    @abstractmethod
    def compute(self, predicted_probabilities, predicted_labels, true_labels):
        """Compute the calibration metric.

        This method must be implemented by subclasses.

        Args:
            predicted_probabilities: The predicted probabilities.
            predicted_labels: The predicted labels.
            true_labels: The true labels.
        """
        pass

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

    @staticmethod
    def validate(predicted_probabilities, true_labels) -> bool:
        """Validate the input data.

        Args:
            predicted_probabilities: The predicted probabilities.
            true_labels: The true labels.

        Returns:
            bool: True if the lengths of predicted_probabilities and true_labels match, False otherwise.
        """
        return len(predicted_probabilities) == len(true_labels)

    @staticmethod
    def _convert_to_numpy(data):
        """Convert input data to a numpy array.

        Args:
            data: Input data, which can be a list, numpy array, or pandas DataFrame/Series.

        Returns:
            np.ndarray: The input data converted to a numpy array.

        Raises:
            TypeError: If the input data is not a list, numpy array, or pandas DataFrame/Series.
        """
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.to_numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise TypeError("Input must be a list, numpy array, or pandas DataFrame/Series")
