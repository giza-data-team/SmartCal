import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .calibration_base import CalibratorBase
from config.configuration_manager import ConfigurationManager
from utils.timer import time_operation

config_manager = ConfigurationManager()


class AdaptiveTemperatureScalingCalibrator(CalibratorBase):
    """
    Implements Adaptive Temperature Scaling for multi-class calibration with multiple modes.
    Supports linear (confidence-based), entropy-based, or hybrid temperature scaling.

    Modes:
        - 'linear': Uses confidence levels to determine temperature scaling
        - 'entropy': Uses prediction entropy to determine temperature scaling
        - 'hybrid': Uses both confidence and entropy (default)
    """

    def __init__(self, mode='hybrid', confidence_bins=config_manager.n_confidence_bins, entropy_bins=config_manager.n_entropy_bins,
                 initial_T=config_manager.initial_T,
                 lr_tempscaling=config_manager.lr_tempscaling,
                 max_iter_tempscaling=config_manager.max_iter_tempscaling,
                 seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed)

        if mode not in ['linear', 'entropy', 'hybrid']:
            raise ValueError("Mode must be one of 'linear', 'entropy', or 'hybrid'")

        self.mode = mode
        self.confidence_bins = confidence_bins
        self.entropy_bins = entropy_bins
        self.lr_tempscaling = lr_tempscaling
        self.max_iter_tempscaling = max_iter_tempscaling
        self.initial_T = initial_T

        # Initialize temperature parameters based on mode
        self._initialize_temperatures()

        self.confidence_boundaries = None
        self.entropy_boundaries = None
        self.fitted = False
        self.timing = {}

    def _initialize_temperatures(self):
        """Initialize temperature parameters based on the selected mode"""
        if self.mode == 'linear':
            self.temperatures = nn.Parameter(torch.ones(self.confidence_bins) * self.initial_T)
        elif self.mode == 'entropy':
            self.temperatures = nn.Parameter(torch.ones(self.entropy_bins) * self.initial_T)
        else:  # hybrid
            self.temperatures = nn.Parameter(torch.ones(self.confidence_bins, self.entropy_bins) * self.initial_T)

    def _compute_entropy(self, probs):
        """Compute entropy of probability distributions"""
        log_probs = torch.log(torch.clamp(probs, min=1e-10))
        entropy = -torch.sum(probs * log_probs, dim=1)
        return entropy

    def _determine_boundaries(self, confidences=None, entropies=None):
        """Determine bin boundaries based on the selected mode"""
        if self.mode in ['linear', 'hybrid'] and confidences is not None:
            self.confidence_boundaries = []
            for i in range(1, self.confidence_bins):
                self.confidence_boundaries.append(np.percentile(confidences, i * 100 / self.confidence_bins))

        if self.mode in ['entropy', 'hybrid'] and entropies is not None:
            self.entropy_boundaries = []
            for i in range(1, self.entropy_bins):
                self.entropy_boundaries.append(np.percentile(entropies, i * 100 / self.entropy_bins))

    def _get_bin_indices(self, confidences=None, entropies=None):
        """Get bin indices based on the selected mode"""
        if self.mode == 'linear':
            conf_indices = torch.zeros(confidences.shape[0], dtype=torch.long)
            for i, boundary in enumerate(self.confidence_boundaries):
                conf_indices += (confidences > boundary).long()
            return conf_indices

        elif self.mode == 'entropy':
            ent_indices = torch.zeros(entropies.shape[0], dtype=torch.long)
            for i, boundary in enumerate(self.entropy_boundaries):
                ent_indices += (entropies > boundary).long()
            return ent_indices

        else:  # hybrid
            conf_indices = torch.zeros(confidences.shape[0], dtype=torch.long)
            for i, boundary in enumerate(self.confidence_boundaries):
                conf_indices += (confidences > boundary).long()

            ent_indices = torch.zeros(entropies.shape[0], dtype=torch.long)
            for i, boundary in enumerate(self.entropy_boundaries):
                ent_indices += (entropies > boundary).long()

            return conf_indices, ent_indices

    def _get_temperature_values(self, bin_indices, batch_size):
        """Get temperature values based on bin indices and mode"""
        if self.mode == 'linear' or self.mode == 'entropy':
            return self.temperatures[bin_indices]
        else:  # hybrid
            conf_indices, ent_indices = bin_indices
            return self.temperatures[conf_indices, ent_indices]

    @time_operation
    def fit(self, predictions, ground_truth):
        """
        Trains the adaptive temperature scaling model.

        Args:
            predictions (np.ndarray): Model logits of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        """
        self.set_random_seed(self.seed)  # Ensure full determinism

        predictions, ground_truth = self.validate_inputs(predictions, ground_truth)
        predictions = torch.tensor(predictions, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)

        # Reset temperature parameters before training
        self._initialize_temperatures()

        # Compute initial softmax probabilities
        probs = torch.nn.functional.softmax(predictions, dim=1)
        confidences = None
        entropies = None

        if self.mode in ['linear', 'hybrid']:
            confidences, _ = torch.max(probs, dim=1)

        if self.mode in ['entropy', 'hybrid']:
            entropies = self._compute_entropy(probs)

        # Determine bin boundaries
        self._determine_boundaries(
            confidences.detach().numpy() if confidences is not None else None,
            entropies.detach().numpy() if entropies is not None else None
        )

        optimizer = optim.LBFGS([self.temperatures], lr=self.lr_tempscaling, max_iter=self.max_iter_tempscaling)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()

            # Get bin indices based on mode
            if self.mode == 'linear':
                bin_indices = self._get_bin_indices(confidences)
            elif self.mode == 'entropy':
                bin_indices = self._get_bin_indices(entropies=entropies)
            else:  # hybrid
                bin_indices = self._get_bin_indices(confidences, entropies)

            # Get temperature values
            temp_values = self._get_temperature_values(bin_indices, predictions.shape[0])

            # Apply temperature scaling
            scaled_logits = predictions / torch.clamp(temp_values.unsqueeze(1), min=1e-6)
            loss = criterion(torch.nn.functional.log_softmax(scaled_logits, dim=1), ground_truth)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.fitted = True

        # Prepare metadata
        self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": predictions.shape[1],
        })

        params_dict = {
            "mode": self.mode,
            "initial_T": self.initial_T,
            "lr_tempscaling": self.lr_tempscaling,
            "max_iter_tempscaling": self.max_iter_tempscaling,
        }

        if self.mode in ['linear', 'hybrid']:
            params_dict["confidence_bins"] = self.confidence_bins
            params_dict["confidence_boundaries"] = [float(b) for b in self.confidence_boundaries]

        if self.mode in ['entropy', 'hybrid']:
            params_dict["entropy_bins"] = self.entropy_bins
            params_dict["entropy_boundaries"] = [float(b) for b in self.entropy_boundaries]

        # Store optimized temperatures
        if self.mode == 'linear' or self.mode == 'entropy':
            params_dict["optimized_temperatures"] = [round(t.item(), 6) for t in self.temperatures]
        else:  # hybrid
            params_dict["optimized_temperatures"] = [[round(t.item(), 6) for t in row] for row in self.temperatures]

        self.metadata["params"].update(params_dict)

    @time_operation
    def predict(self, test_data):
        """
        Applies adaptive temperature scaling to logits.

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

        # Get initial probabilities
        probs = torch.nn.functional.softmax(test_data, dim=1)
        confidences = None
        entropies = None

        if self.mode in ['linear', 'hybrid']:
            confidences, _ = torch.max(probs, dim=1)

        if self.mode in ['entropy', 'hybrid']:
            entropies = self._compute_entropy(probs)

        # Get bin indices based on mode
        if self.mode == 'linear':
            bin_indices = self._get_bin_indices(confidences)
        elif self.mode == 'entropy':
            bin_indices = self._get_bin_indices(entropies=entropies)
        else:  # hybrid
            bin_indices = self._get_bin_indices(confidences, entropies)

        # Get temperature values
        temp_values = self._get_temperature_values(bin_indices, test_data.shape[0])

        # Apply temperature scaling
        scaled_logits = test_data / torch.clamp(temp_values.unsqueeze(1), min=1e-6)
        return torch.nn.functional.softmax(scaled_logits, dim=1).detach().numpy()

    def get_timing(self):
        return self.timing