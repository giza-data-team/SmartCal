import numpy as np


def convert_one_hot_to_labels(y_true: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoded labels to single-label format.

    :param y_true: A NumPy array containing ground truth labels (one-hot or single-label).
    :return: A NumPy array with single-label format.
    """
    if y_true.ndim == 2 and y_true.shape[1] > 1:  # Detect one-hot encoding
        return np.argmax(y_true, axis=1)  # Convert to single-label format
    return y_true  # Return unchanged if already single-label

def normalize_probabilities(probs, tolerance=0):
    """
    Normalize a 1D or 2D array of prediction probabilities so that each row sums to 1.
    If the input is already normalized (within a small tolerance), it is returned as is.

    Args:
        probs (list or np.array): A 1D or 2D array of prediction probabilities.
        tolerance (float): Allowed deviation from 1.0 to consider the input already normalized.
        
    Returns:
        np.array: Normalized probabilities if needed, else original input.
    """
    probs = np.array(probs, dtype=np.float64)  # Convert to NumPy array
    
    if np.any(probs < 0):  # Ensure no negative values
        raise ValueError("Probabilities must be non-negative.")
    
    total = np.sum(probs, axis=-1, keepdims=True)  # Sum along the last axis
    
    # Check if already normalized within tolerance
    if np.all(np.abs(total - 1) < tolerance):
        return probs  # Return as is if already normalized
    
    if np.any(total == 0):  # Prevent division by zero
        raise ValueError("Sum of probabilities cannot be zero for any row.")
    
    return probs / total  # Normalize only if needed