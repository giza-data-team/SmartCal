import numpy as np
from betacal import BetaCalibration as BetaCal
from calibration_algorithms.calibration_base import CalibratorBase
from config.configuration_manager.configuration_manager import ConfigurationManager
from utils.timer import time_operation

config = ConfigurationManager()

class BetaCalibrator(CalibratorBase):
    """
    Wrapper that uses the 'betacal' library for Beta Calibration,
    handling both binary and multi-class (one-vs-rest).

    model_type
    ----------
    model_type : str (default="abm")
        Which Beta model from betacal to use: "abm", "am", or "ab".

    Attributes
    ----------
    calibrators_ : dict
        For binary classification, a single entry calibrators_[1].
        For multi-class, calibrators_[class_idx] -> BetaCalibration instance.
    n_classes_ : int
        Number of classes seen during fit.
    fitted : bool
        Whether the calibrator has been fit.
    """

    def __init__(self, model_type=config.beta_model, seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed=seed)
        
        self.model_type = model_type
        self.calibrators_ = {}
        self.n_classes_ = None
        self.timing = {} 
        self.metadata.update({
            "calibration_type": "beta"
        })
        self.metadata["params"].update({
            "beta_model_type": model_type
        })

    @time_operation
    def fit(self, probabilities, labels):
        """
        Fit Beta calibration. If n_classes=2 => calibrate only the 'positive' column.
        Otherwise do one-vs-rest for multi-class, then re-normalize at predict time.

        model_type
        ----------
        probabilities : np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities for each class.
        labels : array-like, shape (n_samples,)
            True labels (0..n_classes-1).
        """
        probabilities, labels = super().validate_inputs(probabilities, labels)

        n_samples, n_classes = probabilities.shape
        if len(labels) != n_samples:
            raise ValueError("Labels length must match number of probability rows.")

        self.n_classes_ = n_classes
        self.metadata["dataset_info"].update({
            "n_samples": n_samples,
            "n_classes": n_classes,
        })
        
        self.logger.info(
            f"BetaCalibrator.fit started with {n_samples} samples, {n_classes} classes, mode={self.model_type}")

        # --- Binary case: only calibrate column=1 ---
        if n_classes == 2:
            p_pos = probabilities[:, 1]
            y_bin = (labels == 1).astype(int)  # binary 0/1
            cal = BetaCal(parameters=self.model_type)
            cal.fit(p_pos, y_bin)
            self.calibrators_[1] = cal

        else:
            # --- Multi-class: one-vs-rest for each class ---
            for c in range(n_classes):
                p_c = probabilities[:, c]
                y_c = (labels == c).astype(int)
                cal = BetaCal(parameters=self.model_type)
                cal.fit(p_c, y_c)
                self.calibrators_[c] = cal

        self.fitted = True
        self.logger.info(f"Fitted BetaCalibrator with {len(self.calibrators_)} calibrators")

        return self

    @time_operation
    def predict(self, probabilities):
        """
        Predict calibrated probabilities.

        model_type
        ----------
        probabilities : np.ndarray, shape (n_samples, n_classes)
            Uncalibrated probabilities to calibrate.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            The calibrated probabilities, row-wise summing to 1.
        """
        if not self.fitted:
            raise RuntimeError("BetaCalibrator not fitted. Call `fit` first.")

        probabilities = np.asarray(probabilities)
        if probabilities.shape[1] != self.n_classes_:
            raise ValueError("Number of columns in `probabilities` differs from training.")

        n_samples, n_classes = probabilities.shape
        out = np.zeros_like(probabilities, dtype=float)

        # Binary => only calibrate the positive column => column=1
        if n_classes == 2:
            cal = self.calibrators_[1]
            p_pos = cal.predict(probabilities[:, 1])  # shape (n_samples,)
            out[:, 1] = p_pos
            out[:, 0] = 1.0 - p_pos
            return out

        # Multi-class => one-vs-rest + re-normalize
        for c in range(n_classes):
            cal = self.calibrators_[c]
            p_c = probabilities[:, c]
            out[:, c] = cal.predict(p_c)

        # Re-normalize so each row sums to 1
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, 1e-15, None)
        out /= row_sums

        return out
        
    def get_timing(self):
        """Return the timing information for fit and predict operations."""
        return self.timing