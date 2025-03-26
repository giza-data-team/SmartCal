from sklearn.isotonic import IsotonicRegression
from calibration_algorithms.calibration_base import CalibratorBase
import numpy as np
from utils.timer import time_operation

class IsotonicCalibrator(CalibratorBase):
    """
    Isotonic calibration for both binary and multi-class classification.
    Binary: Only calibrates the probability for the positive class.
    Multi-class: One-vs-rest approach, then re-normalizes probabilities.
    """

    def __init__(self, seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed=seed)
        self.models = {}        # dict: class_idx -> IsotonicRegression instance
        self.n_classes_ = None
        self.timing = {} 
        self.metadata.update({
            "calibration_type": "isotonic",
        })

    @time_operation
    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        predictions : shape (n_samples, n_classes)
            Assumed to be probabilities, not raw logits.

        ground_truth : shape (n_samples,)
            Class labels.
        """
        predictions, ground_truth = super().validate_inputs(predictions, ground_truth)

        self.n_classes_ = predictions.shape[1]
        n_samples = predictions.shape[0]
        self.metadata["dataset_info"].update({
            "n_samples": n_samples,
            "n_classes": self.n_classes_
        })
        
        self.logger.info(f"IsotonicCalibration.fit with {n_samples} samples, {self.n_classes_} classes")

        # For binary classification: we only calibrate the 'positive' column
        # For multi-class: do one-vs-rest and re-normalize
        if self.n_classes_ == 2:
            # We consider class '1' as "positive"
            # Extract the probability for the positive class
            p_pos = predictions[:, 1]
            # Convert ground_truth to a binary [0,1] vector
            bin_labels = (ground_truth == 1).astype(float)

            # Create and fit an isotonic regression model
            iso_model = IsotonicRegression(out_of_bounds='clip')
            iso_model.fit(p_pos, bin_labels)
            self.models[1] = iso_model

        else:
            # Multi-class: one vs rest for each class
            for class_idx in range(self.n_classes_):
                bin_labels = (ground_truth == class_idx).astype(float)
                # Probability for "this" class
                p_class = predictions[:, class_idx]

                iso_model = IsotonicRegression(out_of_bounds='clip')
                iso_model.fit(p_class, bin_labels)
                self.models[class_idx] = iso_model

        self.fitted = True
        self.metadata["params"]["n_models"] = len(self.models)
        self.logger.info(f"IsotonicCalibration fitted {len(self.models)} model(s)")

    @time_operation
    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        test_data : shape (n_samples, n_classes)
            Probabilities to be calibrated (not raw logits).
        Returns
        -------
        calibrated_probs : shape (n_samples, n_classes)
        """
        if not self.fitted:
            raise RuntimeError("Calibrator not fitted. Call `fit` first.")

        if test_data.shape[1] != self.n_classes_:
            raise ValueError("Number of columns in `test_data` must match training.")

        n_samples = test_data.shape[0]
        calibrated_probs = np.zeros_like(test_data)

        if self.n_classes_ == 2:
            # For binary, we only have an Isotonic model for the 'positive' class
            iso_model = self.models[1]
            p_pos = test_data[:, 1]
            calibrated_pos = iso_model.predict(p_pos)
            calibrated_neg = 1.0 - calibrated_pos

            calibrated_probs[:, 0] = calibrated_neg
            calibrated_probs[:, 1] = calibrated_pos

        else:
            # Multi-class: apply each isotonic model, then re-normalize
            for class_idx in range(self.n_classes_):
                iso_model = self.models[class_idx]
                p_class = test_data[:, class_idx]
                calibrated_probs[:, class_idx] = iso_model.predict(p_class)

            # renormalize row-wise so they sum to 1
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            # avoid division by zero
            np.clip(row_sums, 1e-15, None, out=row_sums)
            calibrated_probs /= row_sums

        return calibrated_probs
    
    def get_timing(self):
        """Return the timing information for fit and predict operations."""
        return self.timing
