import numpy as np
import logging

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


logger = logging.getLogger(__name__)
config_manager = ConfigurationManager()

def calculate_calibration_curve(y_true, y_prob, n_bins = config_manager.n_bins_calibraion_curve):
    """
    Calculate calibration curve data for multiclass classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (can be binary or multiclass)
    y_prob : array-like
        Predicted probabilities for all classes
        Shape should be (n_samples, n_classes)
    n_bins : int
        Number of bins for the calibration curve
        
    Returns:
    --------
    tuple or None
        - mean_predicted_probs: array of mean predicted probabilities per bin
        - true_probs: array of true probabilities per bin
        - bin_counts: array of sample counts per bin
        Returns None if inputs are invalid.
    """
    try:
        y_true = np.array(y_true).astype(int)
        y_prob = np.array(y_prob)
        
        # Check for empty inputs
        if y_true.size == 0 or y_prob.size == 0:
            return None
        
        # Handle binary case
        if y_prob.ndim == 1:
            y_prob = np.vstack([1 - y_prob, y_prob]).T
            
        # Dimension validation
        if len(y_true) != len(y_prob):
            raise ValueError("Number of samples in y_true and y_prob must match")
         
        # For multiclass case validate class labels
        if y_prob.shape[1] > 2:  # More than binary classification
            unique_labels = np.unique(y_true)
            if not (np.all(np.isin(unique_labels, np.arange(y_prob.shape[1])))):
                raise ValueError("Class labels in y_true must be integers in the range [0, n_classes-1].")
              
        # Check for invalid number of bins
        if n_bins <= 0:
            raise ValueError("Number of bins must be positive and higher than 0")
        
        n_classes = y_prob.shape[1]
        
        # Initialize arrays to store results
        mean_predicted_probs = np.zeros((n_bins, n_classes))
        true_probs = np.zeros((n_bins, n_classes))
        bin_counts = np.zeros((n_bins, n_classes))
        
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        
        # Calculate calibration curve for each class
        for class_idx in range(n_classes):
            class_confidences = y_prob[:, class_idx]
            class_true_labels = (y_true == class_idx).astype(int)  # True label for the current class
            bin_indices = np.digitize(class_confidences, bins) - 1
            
            for bin_idx in range(n_bins):
                mask = bin_indices == bin_idx
                # Handle empty bins 
                if np.sum(mask) > 0:
                    bin_counts[bin_idx, class_idx] = np.sum(mask)
                    mean_predicted_probs[bin_idx, class_idx] = np.mean(class_confidences[mask])
                    true_probs[bin_idx, class_idx] = np.mean(class_true_labels[mask])
                else:
                    # Fill empty bins with 0 instead of NaN
                    mean_predicted_probs[bin_idx, class_idx] = 0.0
                    true_probs[bin_idx, class_idx] = 0.0       
            # Log the results for the current class
            logger.info(f"\nCalibration Results for Class {class_idx} ({n_bins} bins):")
            for i in range(n_bins):
                if not np.isnan(mean_predicted_probs[i, class_idx]):
                    logger.info(f"Bin {i+1}: pred={mean_predicted_probs[i, class_idx]:.3f}, true={true_probs[i, class_idx]:.3f}, n={int(bin_counts[i, class_idx])}")
        
        return mean_predicted_probs, true_probs, bin_counts
        
    except Exception as e:
        logger.error(f"Error in multiclass calibration curve calculation: {str(e)}")
        raise