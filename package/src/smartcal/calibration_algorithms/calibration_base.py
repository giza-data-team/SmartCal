import random
import numpy as np
import torch
import logging
from abc import ABC, abstractmethod

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


logging.basicConfig(level=logging.INFO, force=True)

class CalibratorBase(ABC):
    """
    Abstract base class for calibration methods.
    Provides a standardized interface for calibration models.
    """
    
    # Load default configurations
    config_manager = ConfigurationManager()
    DEFAULT_SEED = config_manager.random_seed  # Default random seed
    
    def __init__(self, seed=DEFAULT_SEED):
        self.fitted = False
        self.seed = seed
        self.set_random_seed(seed)
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logs = True
        
        # Initialize metadata dictionary
        self.metadata = {
            "calibration_type": self.__class__.__name__.lower().replace('calibrator', ''),
            "params": {},
            "dataset_info": {}
        }
    
    def set_random_seed(self, seed: int):
        """
        Sets a fixed random seed for reproducibility.

        Args:
            seed (int): The random seed value.
        """
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, got {type(seed)}.")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Fits the calibration model.
        
        Args:
            predictions (np.ndarray): Model output probabilities or logits of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        """
        pass
    
    @abstractmethod
    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generates calibrated probabilities.
        
        Args:
            test_data (np.ndarray): Uncalibrated model outputs of shape (n_samples, n_classes).
        
        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).
        """
        pass
    
    def validate_inputs(self, predictions, ground_truth=None):
        """
        Validates input shapes and types.
        
        Args:
            predictions (np.ndarray): Model output logits or probabilities.
            ground_truth (np.ndarray): True labels.
        
        Returns:
            tuple: Validated predictions and ground truth labels.
        
        Raises:
            ValueError: If inputs do not match expected formats.
        """
        predictions = np.asarray(predictions)
        ground_truth = np.asarray(ground_truth)
        
        if predictions.ndim != 2 or ground_truth.ndim != 1:
            raise ValueError("Invalid input shapes: Predictions must be (n_samples, n_classes), labels must be (n_samples,).")
        
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Predictions must be finite real numbers.")
        
        if ground_truth.dtype.kind not in {'i', 'u'}:
            raise ValueError("Ground truth labels must be integers.")
        
        if predictions.shape[0] != ground_truth.shape[0]:
            raise ValueError("Mismatched dimensions: predictions and labels must have the same number of samples.")
        
        return predictions, ground_truth
    