import numpy as np
from scipy.optimize import minimize

from smartcal.calibration_algorithms.calibration_base import CalibratorBase
from smartcal.utils.timer import time_operation


class PlattScalingCalibrator(CalibratorBase):
    def __init__(self, seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed=seed)
        self.coefficients = {}
        self.n_classes_ = None
        self.timing = {} 
        self.metadata.update({
            "calibration_type": "platt_scaling",
        })
  
    def _fit_one_class(self, logits: np.ndarray, binary_labels: np.ndarray):
        """Fit Platt scaling for a single class on log-odds."""

        def loss(params, reg_strength=1.0):
            w, b = params
            p = 1.0 / (1.0 + np.exp(-(w * logits + b)))
            p = np.clip(p, 1e-15, 1 - 1e-15)
            # Negative log-likelihood
            nll = -np.sum(binary_labels * np.log(p) + (1 - binary_labels) * np.log(1 - p))
            # L2 penalty on w (not usually on b)
            penalty = 0.5 * reg_strength * (w ** 2)
            return nll + penalty

        initial_params = [1.0, 0.0]
        result = minimize(loss, initial_params, method='L-BFGS-B')
        if not result.success:
            self.logger.warning(f"Optimization failed: {result.message}")
            return initial_params  # fallback
        return result.x

    @time_operation
    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Fit Platt scaling parameters. We treat `predictions` as probabilities, so we
        convert them to log-odds. For binary classification, we only calibrate the
        "positive" column. For multi-class, we do one-vs-rest.
        """
        predictions, ground_truth = super().validate_inputs(predictions, ground_truth)

        self.n_classes_ = predictions.shape[1]
        self.coefficients = {}

        # Log dataset info
        self.metadata["dataset_info"].update({
            "n_samples": len(predictions),
            "n_classes": self.n_classes_
        })
        self.logger.info(f"PlattScaling.fit started with {len(predictions)} samples, {self.n_classes_} classes")

        # === Binary special case: calibrate only the positive column. ===
        if self.n_classes_ == 2:
            # The "positive" column is index 1
            p = np.clip(predictions[:, 1], 1e-15, 1 - 1e-15)
            logits = np.log(p / (1 - p))

            # Binary labels for the "positive" class
            binary_labels = (ground_truth == 1).astype(int)

            self.coefficients[1] = self._fit_one_class(logits, binary_labels)

        else:
            # === Multi-class: one-vs-rest approach ===
            for class_idx in range(self.n_classes_):
                binary_labels = (ground_truth == class_idx).astype(int)

                p = np.clip(predictions[:, class_idx], 1e-15, 1 - 1e-15)
                logits = np.log(p / (1.0 - p))

                self.coefficients[class_idx] = self._fit_one_class(logits, binary_labels)

        self.fitted = True

        # Store coefficients
        self.metadata["params"]["coefficients"] = {str(k): list(v) for k, v in self.coefficients.items()}
        self.logger.info(f"PlattScaling fitted coefficients: {self.metadata['params']['coefficients']}")

    @time_operation
    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to test data, assumed to be probabilities. For binary,
        we only calibrate the positive column. For multi-class, we calibrate
        each column (one-vs-rest) and then re-normalize.
        """
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted. Call `fit` first.")

        n_samples, n_cols = test_data.shape
        if n_cols != self.n_classes_:
            raise ValueError("Number of columns in `test_data` differs from training phase.")

        # === Binary special case ===
        if self.n_classes_ == 2:
            calibrated_probs = np.zeros((n_samples, 2))
            w, b = self.coefficients[1]

            # Convert the positive column to log-odds
            p = np.clip(test_data[:, 1], 1e-15, 1 - 1e-15)
            logits = np.log(p / (1 - p))

            # Sigmoid(w * logit + b)
            pos_calibrated = 1.0 / (1.0 + np.exp(-(w * logits + b)))
            neg_calibrated = 1.0 - pos_calibrated

            calibrated_probs[:, 0] = neg_calibrated
            calibrated_probs[:, 1] = pos_calibrated
            return calibrated_probs

        else:
            # === Multi-class scenario: one-vs-rest + re-normalize ===
            calibrated_probs = np.zeros((n_samples, self.n_classes_))
            for class_idx in range(self.n_classes_):
                w, b = self.coefficients[class_idx]
                p = np.clip(test_data[:, class_idx], 1e-15, 1 - 1e-15)
                logits = np.log(p / (1.0 - p))
                calibrated_probs[:, class_idx] = 1.0 / (1.0 + np.exp(-(w * logits + b)))

            # Re-normalize row-wise
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            calibrated_probs /= np.clip(row_sums, 1e-15, None)
            return calibrated_probs
               
    def get_timing(self):
        """Return the timing information for fit and predict operations."""
        return self.timing

    def get_timing(self):
        return self.timing