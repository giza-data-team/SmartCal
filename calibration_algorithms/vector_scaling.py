import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .calibration_base import CalibratorBase
from config.configuration_manager import ConfigurationManager
from utils.timer import time_operation

config_manager = ConfigurationManager()


class VectorScalingCalibrator(CalibratorBase):
    """
    Implements Vector Scaling calibration.
    Applies a learned affine transformation Z⋅ W + b to logits before softmax.
    """

    def __init__(self, lr=config_manager.lr_vectorscaling, max_iter=config_manager.max_iter_vectorscaling,
                 seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed)

        self.lr = lr
        self.max_iter = max_iter
        self.W = None  # Element-wise scaling vector (C)
        self.b = None  # Bias vector (C)
        self.fitted = False
        self.timing = {}

    @time_operation
    def fit(self, predictions, ground_truth):
        """
        Fits the vector scaling model using LBFGS optimization.

        Args:
            predictions (np.ndarray): Model logits of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        """
        self.set_random_seed(self.seed)

        predictions, ground_truth = self.validate_inputs(predictions, ground_truth)
        n_classes = predictions.shape[1]

        predictions = torch.tensor(predictions, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)

        # Initialize W as identity and b as zero
        self.W = nn.Parameter(torch.ones(n_classes, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(n_classes, dtype=torch.float32))

        optimizer = optim.LBFGS([self.W, self.b], lr=self.lr, max_iter=self.max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = predictions * self.W + self.b
            loss = criterion(torch.nn.functional.log_softmax(scaled_logits, dim=1), ground_truth)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.fitted = True

        # Store optimized values
        self.metadata["params"].update({
            "optimized_W": self.W.detach().numpy().tolist(),
            "optimized_b": self.b.detach().numpy().tolist()
        })
        self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": predictions.shape[1],
        })

    @time_operation
    def predict(self, test_data):
        """
        Applies vector scaling to test logits.

        Args:
            test_data (np.ndarray): Logits of shape (n_samples, n_classes).

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        if not isinstance(test_data, np.ndarray):
            raise ValueError(f"Expected input to be a NumPy array, but got {type(test_data)}.")

        if test_data.ndim != 2:
            raise ValueError(f"Expected input shape (N, C), but got {test_data.shape}.")

        if test_data.shape[1] != self.W.shape[0]:
            raise ValueError(f"Expected input shape (N, {self.W.shape[0]}), but got {test_data.shape}.")

        test_data = torch.tensor(test_data, dtype=torch.float32)

        scaled_logits = test_data * self.W + self.b
        return torch.nn.functional.softmax(scaled_logits, dim=1).detach().numpy()

    def get_timing(self):
        return self.timing
