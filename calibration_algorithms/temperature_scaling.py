import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .calibration_base import CalibratorBase
from config.configuration_manager import ConfigurationManager
from utils.timer import time_operation


config_manager = ConfigurationManager()

class TemperatureScalingCalibrator(CalibratorBase):
    """
    Implements Temperature Scaling for multi-class calibration.
    Adjusts model confidence using a single temperature parameter.
    """
    def __init__(self, initial_T=config_manager.initial_T, lr_tempscaling=config_manager.lr_tempscaling, max_iter_tempscaling=config_manager.max_iter_tempscaling, seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed)

        # Use provided values or default to ConfigurationManager values
        self.lr_tempscaling = lr_tempscaling 
        self.max_iter_tempscaling = max_iter_tempscaling
        self.initial_T=initial_T
        self.temperature = nn.Parameter(torch.ones(1) * self.initial_T)  # Initialize T=1.0
        self.optimized_temperature = None  # Store optimized temperature value
        self.timing = {}
    
    @time_operation
    def fit(self, predictions, ground_truth):
        """
        Trains the temperature scaling model.
        
        Args:
            predictions (np.ndarray): Model logits of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        """
        self.set_random_seed(self.seed)  # Ensure full determinism

        predictions, ground_truth = self.validate_inputs(predictions, ground_truth)
        predictions = torch.tensor(predictions, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)

        # Reset temperature parameter before training
        self.temperature = nn.Parameter(torch.ones(1) * self.initial_T, requires_grad=True)
        
        optimizer = optim.LBFGS([self.temperature], lr=self.lr_tempscaling, max_iter=self.max_iter_tempscaling)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = predictions / torch.clamp(self.temperature, min=1e-6)  # Avoid division by zero
            loss = criterion(torch.nn.functional.log_softmax(scaled_logits, dim=1), ground_truth)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.fitted = True
        self.optimized_temperature = round(self.temperature.item(), 6)  # Ensure consistent precision

        self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": predictions.shape[1],
        })
        self.metadata["params"].update({
            "initial_T": self.initial_T,
            "lr_tempscaling": self.lr_tempscaling,
            "max_iter_tempscaling": self.max_iter_tempscaling,
            "optimized_temperature": self.optimized_temperature
        })
    
    @time_operation
    def predict(self, test_data):
        """
        Applies temperature scaling to logits.

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

        test_data = torch.tensor(test_data, dtype=torch.float32)
        return torch.nn.functional.softmax(test_data / torch.clamp(self.temperature, min=1e-6), dim=1).detach().numpy()
    
    def get_timing(self):
        return self.timing