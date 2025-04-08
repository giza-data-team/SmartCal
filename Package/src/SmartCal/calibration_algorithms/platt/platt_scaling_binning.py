import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer

from ..calibration_base import CalibratorBase
from SmartCal.config.configuration_manager import ConfigurationManager
from SmartCal.utils.timer import time_operation  # For measuring execution time


config_manager = ConfigurationManager()

class PlattBinnerScalingCalibrator(CalibratorBase):
    def __init__(self, num_bins=config_manager.n_bin_cal, seed=CalibratorBase.DEFAULT_SEED):
        """
        Initialize the Platt Binner Scaling calibrator.
        :param num_bins: Number of bins to use in the binning step.
        """
        super().__init__(seed)
        self.num_bins = num_bins
        self.platt_models = None
        self.binners = None
        self.bin_means = None
        self.n_classes = None
        self.metadata.update({"calibration_type": "platt_binner_scaling"})  # Store calibration type
        self.metadata["params"].update({"num_bins": self.num_bins})  
        self.timing = {}


    @time_operation
    def fit(self, logits, labels):
        """
        Fit the Platt Scaling model and apply binning for multi-class calibration.
        :param logits: (n_samples, n_classes) array of logits (pre-softmax scores).
        :param labels: (n_samples,) array of true labels.
        """
        logits, labels = super().validate_inputs(logits, labels)  # Ensure input consistency
        logits = np.array(logits)
        labels = np.array(labels)
        n_samples, self.n_classes = logits.shape

        self.platt_models = []
        self.binners = []
        self.bin_means = []

        # Log dataset info
        self.metadata["dataset_info"].update({"n_samples": n_samples, "n_classes": self.n_classes})

        # One-vs-All Platt Scaling + Binning
        for class_idx in range(self.n_classes):
            # Convert labels to binary: is this class or not?
            binary_labels = (labels == class_idx).astype(int)

            # Train logistic regression for Platt Scaling
            model = LogisticRegression(solver="lbfgs")
            model.fit(logits[:, class_idx].reshape(-1, 1), binary_labels)
            self.platt_models.append(model)

            # Compute Platt-scaled probabilities
            platt_probs = model.predict_proba(logits[:, class_idx].reshape(-1, 1))[:, 1]

            # Train binning model (Equal-Width Bins)
            try:
                binner = KBinsDiscretizer(n_bins=self.num_bins, encode="ordinal", strategy="uniform")
                bin_indices = binner.fit_transform(platt_probs.reshape(-1, 1)).astype(int).flatten()
            except ValueError as e:
                self.logger.error(f"Binning failed for class {class_idx}: {e}")
                raise RuntimeError("Binning encountered an issue. Try reducing `num_bins`.")

            # Compute actual bin means from training data
            bin_means = np.zeros(self.num_bins)
            for b in range(self.num_bins):
                in_bin = platt_probs[bin_indices == b]
                bin_means[b] = np.mean(in_bin) if len(in_bin) > 0 else 0.5  # Default to 0.5 if bin is empty

            self.binners.append(binner)
            self.bin_means.append(bin_means)

        self.fitted = True

    @time_operation
    def predict(self, logits):
        """
        Apply Platt Scaling + Binning to obtain calibrated probabilities.
        :param logits: (n_samples, n_classes) array of logits.
        :return: (n_samples, n_classes) array of calibrated probabilities.
        """
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted. Call `fit` first.")

        logits = np.array(logits)

        # Ensure input shape matches the number of classes
        if logits.shape[1] != self.n_classes:
            raise ValueError(f"Mismatch in number of classes: expected {self.n_classes}, got {logits.shape[1]}.")

        n_samples = logits.shape[0]
        calibrated_probs = np.zeros((n_samples, self.n_classes))

        for class_idx in range(self.n_classes):
            # Apply Platt Scaling
            platt_probs = self.platt_models[class_idx].predict_proba(logits[:, class_idx].reshape(-1, 1))[:, 1]

            # Apply Binning Calibration
            try:
                bin_indices = self.binners[class_idx].transform(platt_probs.reshape(-1, 1)).astype(int).flatten()
            except ValueError as e:
                raise RuntimeError(f"Binning failed for class {class_idx}: {e}")

            # Ensure bin indices are within range
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

            # Assign bin means
            calibrated_probs[:, class_idx] = self.bin_means[class_idx][bin_indices]

        # Normalize probabilities to ensure they sum to 1
        row_sums = calibrated_probs.sum(axis=1, keepdims=True)
        calibrated_probs /= np.clip(row_sums, 1e-15, None)  # Avoid division by zero

        return calibrated_probs

    def get_timing(self):
        return self.timing