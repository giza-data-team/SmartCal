import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression

from smartcal.calibration_algorithms.calibration_base import CalibratorBase
from smartcal.calibration_algorithms.temperature_scaling import TemperatureScalingCalibrator
from smartcal.utils.timer import time_operation
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config_manager = ConfigurationManager()

class MetaCalibrator(CalibratorBase):
    """
    A unified calibration model supporting:
    - Miscoverage rate constraint (alpha)
    - Coverage accuracy constraint (acc)
    
    This class calibrates model predictions under a given constraint and improves the reliability of confidence scores.
    """
    
    def __init__(self, calibrator_type=config_manager.meta_default_constraint, alpha=None, acc=None, seed=CalibratorBase.DEFAULT_SEED):
        """
        Initializes the MetaCalibrator.
        
        Args:
            fit_type (str, optional): Type of meta calibration constraint. Should be either 'ALPHA' (for miscoverage rate constraint) or 'ACC' (for coverage accuracy constraint).
            alpha (float, optional): Miscoverage rate constraint. Must be a float in the range (0,1). Default is None.
            acc (float, optional): Coverage accuracy constraint. Must be a float in the range (0,1). Default is None.
            seed (int): Random seed for reproducibility.
        
        Raises:
            TypeError: If fit_type is not a string.
            ValueError: If fit_type is not 'ALPHA' or 'ACC'.
            ValueError: If alpha or acc are not in the valid range (0,1).
        """
        super().__init__(seed)
        
        if not isinstance(calibrator_type, str):
            raise TypeError("fit_type must be a string")
        if calibrator_type not in ['ALPHA', 'ACC']:
            raise ValueError("fit_type must be either 'ALPHA' or 'ACC'")
        if alpha is not None and (not isinstance(alpha, (float, int)) or not (0 < alpha < 1)):
            raise ValueError("alpha must be a float in the range (0,1)")
        if acc is not None and (not isinstance(acc, (float, int)) or not (0 < acc < 1)):
            raise ValueError("acc must be a float in the range (0,1)")
        

        if alpha is not None and acc is not None:
            if calibrator_type == 'ALPHA':
                acc = None
            elif calibrator_type == 'ACC':
                alpha = None

        if calibrator_type == 'ALPHA' and alpha == None:
            alpha = config_manager.meta_alpha

        if calibrator_type == 'ACC' and acc == None:
            acc = config_manager.meta_acc

        self.alpha = alpha
        self.acc = acc
        self.timing = {}
        self.metadata["params"].update({"alpha": alpha, "acc": acc})
    
    @time_operation
    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Fits the calibrator to the provided predictions and ground truth labels.
        
        Args:
            predictions (np.ndarray): Model output logits of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        
        Raises:
            AttributeError: If the calibration method is not properly set.
        """
        predictions, ground_truth = self.validate_inputs(predictions, ground_truth)
        self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": predictions.shape[1],
        })
        if self.alpha is not None:
            self._fit_miscoverage(predictions, ground_truth)
        else:
            self._fit_coverage_acc(predictions, ground_truth)
        self.fitted = True
    
    def _fit_miscoverage(self, xs, ys):
        """
        Fits the calibration model under a miscoverage rate constraint.
        
        Args:
            xs (np.ndarray): Logits of shape (N, K).
            ys (np.ndarray): True labels of shape (N,).
        """
        neg_ind = np.argmax(xs, axis=1) == ys
        xs_neg, ys_neg = xs[neg_ind], ys[neg_ind]
        xs_pos, ys_pos = xs[~neg_ind], ys[~neg_ind]
        # Add minimum size check
        if len(xs_neg) < 1:
            raise ValueError("Not enough negative samples for calibration")
        # Ensure n1 is at least 1
        n1 = max(min(int(len(xs_neg) / 10), 500), 1)
        x1, x2, _, y2 = train_test_split(xs_neg, ys_neg, train_size=n1)
        x2 = np.r_[x2, xs_pos]
        y2 = np.r_[y2, ys_pos]
        
        scores_x1 = entropy(softmax(x1, axis=1), axis=1)
        self.threshold = np.quantile(scores_x1, 1 - self.alpha, method="higher")
        
        scores_x2 = entropy(softmax(x2, axis=1), axis=1)
        cond_ind = scores_x2 < self.threshold
        
        # Instantiate temperature scaling calibration model
        self.base_model = TemperatureScalingCalibrator()
        
        ts_x, ts_y = x2[cond_ind], y2[cond_ind]
        self.base_model.fit(ts_x, ts_y)
    
    def _fit_coverage_acc(self, xs, ys):
        """
        Fits the calibration model under a coverage accuracy constraint.
        
        Args:
            xs (np.ndarray): Logits of shape (N, K).
            ys (np.ndarray): True labels of shape (N,).
        """
        bins = 20
        # Ensure n1 is at least 1
        n1 = max(min(int(len(xs) / 10), 500), 1)

        # Add minimum size check
        if len(xs) < 1:
            raise ValueError("Not enough samples for calibration")

        x1, x2, y1, y2 = train_test_split(xs, ys, train_size=n1)
        x1_pred = np.argmax(x1, axis=1)
        scores_x1 = entropy(softmax(x1, axis=1), axis=1)

        accs, ents = [], []
        cut_points = np.quantile(scores_x1, np.linspace(0, 1, bins + 1))
        for (a, b) in zip(cut_points, cut_points[1:]):
            indices = np.where((scores_x1 > a) & (scores_x1 <= b))[0]
            if len(indices) > 0:
                accs.append(np.mean(y1[indices] == x1_pred[indices]))
                ents.append(np.mean(scores_x1[indices]))
            else:
                accs.append(0)
                ents.append(0)

        accs_avg = np.add.accumulate(accs) / (np.arange(len(accs)) + 1)

        # Ensure 1D arrays for Isotonic Regression
        accs_avg = np.array(accs_avg).flatten()
        ents = np.array(ents).flatten()

        # Instantiate Isotonic Regression
        model_l = IsotonicRegression(increasing=False, out_of_bounds="clip")
        model_l.fit(accs_avg.reshape(-1, 1), ents) # X=accs_avg, y=ents

        self.threshold = model_l.predict([[self.acc]])[0]  # Ensure input is 2D for predict()
        if np.isnan(self.threshold):
            raise ValueError("Coverage accuracy should be increased.")

        scores_x2 = entropy(softmax(x2, axis=1), axis=1)
        cond_ind = scores_x2 < self.threshold

        # Instantiate temperature scaling calibration model
        self.base_model = TemperatureScalingCalibrator()

        ts_x, ts_y = x2[cond_ind], y2[cond_ind]
        self.base_model.fit(ts_x, ts_y)
    
    @time_operation
    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generates calibrated probability predictions.
        
        Args:
            test_data (np.ndarray): Logits of shape (N, K).
        
        Returns:
            np.ndarray: Calibrated probabilities of shape (N, K).
        
        Raises:
            AttributeError: If `fit()` has not been called before prediction.
        """
        if not self.fitted:
            raise AttributeError("Run fit on training set first.")
        scores_X = entropy(softmax(test_data, axis=1), axis=1)
        neg_ind = scores_X < self.threshold
        proba_cal = np.empty_like(test_data)
        proba_cal[neg_ind] = self.base_model.predict(test_data[neg_ind])
        proba_cal[~neg_ind] = 1 / test_data.shape[1]
        return proba_cal
    
    def get_timing(self):
        return self.timing