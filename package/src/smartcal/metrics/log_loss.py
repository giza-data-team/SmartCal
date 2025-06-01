import numpy as np
import logging
from sklearn.metrics import log_loss

logger = logging.getLogger(__name__)

def calculate_log_loss(y_true, y_prob):
    """
    Calculate Log Loss (Cross-Entropy Loss) for classification predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True labels (can be binary or multiclass)
    y_prob : array-like
        Predicted probabilities for all classes
        Shape should be (n_samples, n_classes)
        
    Returns:
    --------
    float or None
        Log Loss score (lower is better)
        Returns None if inputs are invalid.
    """
    try:
        y_true = np.array(y_true).astype(int)
        y_prob = np.array(y_prob)
        
        # Check for empty inputs
        if y_true.size == 0 or y_prob.size == 0:
           raise ValueError("Input arrays cannot be empty")
        
        # Handle binary case
        if y_prob.ndim == 1:
            y_prob = np.vstack([1 - y_prob, y_prob]).T
            
        n_samples, n_classes = y_prob.shape
        
        # Strict dimension validation
        if len(y_true) != n_samples:
            raise ValueError("Number of samples in y_true and y_prob must match")
            
        # For multiclass case, validate class labels
        if y_prob.shape[1] > 2:  # More than binary classification
            unique_labels = np.unique(y_true)
            if not (np.all(np.isin(unique_labels, np.arange(y_prob.shape[1])))):
                raise ValueError("Class labels in y_true must be integers in the range [0, n_classes-1].")
        
        # Calculate log loss using sklearn's implementation
        loss = log_loss(y_true, y_prob)
        
        logger.info(f"Log Loss: {loss:.4f}")

        return loss
        
    except Exception as e:
        logger.error(f"Error in log loss calculation: {str(e)}")
        raise