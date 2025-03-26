import numpy as np
from scipy.special import softmax
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from calibration_algorithms.calibration_base import CalibratorBase
from config.configuration_manager import ConfigurationManager
from utils.timer import time_operation

config_manager = ConfigurationManager()

class ProbabilityTreeCalibrator(CalibratorBase):
    """
    Implements Probability Calibration Trees (PCT) for Multiclass Calibration.
    """
    def __init__(self, max_depth=config_manager.probability_tree_max_depth, min_samples_leaf=config_manager.probability_tree_min_samples_leaf, seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed)
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.logit_models = {}  # Logistic regression models at leaf nodes
        self.global_logit_model = None  # Global fallback model
        self.metadata["params"].update({
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        })
        self.timing = {}
    
    def log_odds_transform(self, probs):
        """Applies log-odds transformation to probability estimates."""
        probs = np.clip(probs, 1e-6, 1 - 1e-6)  # Avoid division by zero
        return np.log(probs / (1 - probs))
    
    @time_operation
    def fit(self, X, logits, y):
        """
        Trains the probability calibration tree.
        
        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features)
            logits (np.ndarray): Uncalibrated probability estimates (n_samples, n_classes)
            y (np.ndarray): True labels (n_samples,)
        """
        X, y = self.validate_inputs(X, y)
        logits = np.asarray(logits)
        
        # Fit the decision tree on original features
        self.tree.fit(X, y)
        
        # Get leaf nodes for each training instance
        leaf_indices = self.tree.apply(X)
        
        # Train a global fallback logistic regression model on all data
        global_X_log_odds = self.log_odds_transform(softmax(logits, axis=1))
        self.global_logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        self.global_logit_model.fit(global_X_log_odds, y)
        
        # Train logistic regression models at each leaf node
        for leaf in np.unique(leaf_indices):
            indices = np.where(leaf_indices == leaf)[0]
            if len(indices) < 2:
                # Use global model if not enough samples in leaf
                self.logit_models[leaf] = self.global_logit_model
                continue
            
            X_leaf = logits[indices]  # Use logits as features for logistic regression
            y_leaf = y[indices]
            
            # Convert logits to log-odds
            X_leaf_log_odds = self.log_odds_transform(softmax(X_leaf, axis=1))
            
            # Fit logistic regression model for calibration
            model = LogisticRegression(solver='lbfgs', max_iter=1000)
            model.fit(X_leaf_log_odds, y_leaf)
            self.logit_models[leaf] = model
        
        self.metadata["dataset_info"].update({
            "n_samples": X.shape[0],
            "n_features": X.shape[1]
        })
        self.fitted = True
    
    @time_operation
    def predict(self, X, logits):
        """
        Calibrates probability estimates using the learned tree and logistic models.
        
        Args:
            X (np.ndarray): Feature matrix (n_samples, n_features)
            logits (np.ndarray): Uncalibrated probability estimates (n_samples, n_classes)
        
        Returns:
            np.ndarray: Calibrated probability estimates (n_samples, n_classes)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        
        X = np.asarray(X)
        logits = np.asarray(logits)
        n_classes = logits.shape[1]
        
        # Get leaf nodes for each test instance
        leaf_indices = self.tree.apply(X)
        calibrated_probs = np.zeros_like(logits)
        
        for i, leaf in enumerate(leaf_indices):
            model = self.logit_models.get(leaf, self.global_logit_model)  # Use leaf model or fallback
            log_odds = self.log_odds_transform(softmax(logits[i].reshape(1, -1), axis=1))
            pred_probs = model.predict_proba(log_odds)
            
            # Ensure shape compatibility with expected number of classes
            if pred_probs.shape[1] < n_classes:
                missing_classes = n_classes - pred_probs.shape[1]
                pred_probs = np.hstack([pred_probs, np.full((1, missing_classes), 1e-6)])
            
            calibrated_probs[i] = softmax(pred_probs)
        
        return calibrated_probs
    
    def get_timing(self):
        return self.timing