from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import numpy as np


def compute_classification_metrics(y_true, y_pred, y_prob=None, zero_division=0):
    """
    Compute log loss, accuracy, macro, micro, and weighted precision, recall, and F1-score.
    
    Args:
        y_true (list or array): True labels
        y_pred (list or array): Predicted labels
        y_prob (list or array, optional): Predicted probabilities for log loss calculation (required for log loss)
    
    Returns:
        dict: Dictionary containing the computed metrics
    """
    metrics = {}
    
    if y_prob is not None:
        # Ensure y_prob is a NumPy array
        y_prob = np.array(y_prob)

        # Extract unique classes from y_true
        unique_classes_true = np.unique(y_true)

        # Ensure y_prob has a valid shape
        if y_prob.ndim == 1:
            raise ValueError("y_prob should have shape (n_samples, n_classes), but has shape:", y_prob.shape)

        # Extract class indices from predictions
        unique_classes_pred = np.arange(y_prob.shape[1])

        # Get all unique classes
        all_classes = np.union1d(unique_classes_true, unique_classes_pred)

        # Compute log_loss with explicitly defined labels
        metrics["loss"] = log_loss(y_true, y_prob, labels=all_classes)
    
    metrics.update({
        'accuracy': accuracy_score(y_true, y_pred),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=zero_division),
        'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=zero_division),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=zero_division),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=zero_division),
        'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=zero_division),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=zero_division),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=zero_division),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=zero_division),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=zero_division)
    })

    return metrics
