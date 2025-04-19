import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from smartcal.calibration_algorithms.calibration_base import CalibratorBase
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.utils.timer import time_operation


config_manager = ConfigurationManager()

class DirichletCalibrator(CalibratorBase):
    """
    Implements Dirichlet Calibration for multi-class calibration.
    Adjusts model confidence using a full-rank transformation matrix and bias term.
    """
    def __init__(self, lr=config_manager.lr_tempscaling, max_iter=config_manager.max_iter_tempscaling, seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed)
        self.lr = lr
        self.max_iter = max_iter
        self.transformation_matrix = None
        self.bias = None
        self.timing = {}
    
    @time_operation
    def fit(self, predictions, ground_truth):
        """
        Trains the Dirichlet calibration model.
        
        Args:
            predictions (np.ndarray): Model logits of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        """
        self.set_random_seed(self.seed)
        predictions, ground_truth = self.validate_inputs(predictions, ground_truth)
        predictions = torch.tensor(predictions, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)

        num_classes = predictions.shape[1]
        self.transformation_matrix = nn.Parameter(torch.eye(num_classes, dtype=torch.float32, requires_grad=True))
        self.bias = nn.Parameter(torch.zeros(num_classes, dtype=torch.float32, requires_grad=True))
        
        optimizer = optim.Adam([self.transformation_matrix, self.bias], lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.max_iter):
            optimizer.zero_grad()
            transformed_logits = torch.matmul(predictions, self.transformation_matrix) + self.bias
            loss = criterion(torch.nn.functional.log_softmax(transformed_logits, dim=1), ground_truth)
            loss.backward()
            optimizer.step()
        
        self.fitted = True
        
        self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": num_classes,
        })
        self.metadata["params"].update({
            "learning_rate": self.lr,
            "max_iterations": self.max_iter,
        })
    
    @time_operation
    def predict(self, test_data):
        """
        Applies Dirichlet calibration to logits.

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

        # **NEW: Shape validation**
        if test_data.shape[1] != self.transformation_matrix.shape[0]:
            raise ValueError(
                f"Mismatch in number of classes: expected {self.transformation_matrix.shape[0]}, but got {test_data.shape[1]}"
            )

        calibrated_logits = torch.matmul(test_data, self.transformation_matrix) + self.bias
        return torch.nn.functional.softmax(calibrated_logits, dim=1).detach().numpy()
    
    def get_timing(self):
        """Return the timing information for fit and predict operations."""
        return self.timing
