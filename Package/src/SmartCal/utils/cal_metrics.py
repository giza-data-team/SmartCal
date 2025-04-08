import numpy as np
from SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from SmartCal.config.enums.calibration_metrics_enum import CalibrationMetricsEnum
from SmartCal.metrics.calibration_curve import calculate_calibration_curve
from SmartCal.metrics.brier_score import calculate_brier_score

# Initialize Configuration Manager and get parameters
config_manager = ConfigurationManager()


import numpy as np

def compute_calibration_metrics(probabilities, predictions, true_labels, metrics_to_compute=None):
    """
    Computes selected calibration metrics.
    
    Args:
        probabilities (array-like): Model prediction probabilities.
        predictions (array-like): Predicted class labels.
        true_labels (array-like): Ground truth labels.
        metrics_to_compute (list, optional): List of metrics to compute. Options:
            ['ece', 'mce', 'conf_ece', 'brier_score', 'calibration_curve']
            Default is all metrics.

    Returns:
        dict: Dictionary containing the requested calibration metrics.
    """
    # Available metrics
    available_metrics = {
        'ece': lambda: ece_calculator.compute(
            predicted_probabilities=probabilities,
            predicted_labels=predictions,
            true_labels=true_labels
        ),
        'mce': lambda: mce_calculator.compute(
            predicted_probabilities=probabilities,
            predicted_labels=predictions,
            true_labels=true_labels
        ),
        'conf_ece': lambda: tuple(
            calculator.compute(
                predicted_probabilities=probabilities,
                predicted_labels=predictions,
                true_labels=true_labels
            )
            for calculator in conf_ece_calculators.values()
        ),
        'brier_score': lambda: calculate_brier_score(
            y_true=true_labels,
            y_prob=probabilities
        ),
        'calibration_curve': lambda: calculate_calibration_curve(
            y_true=true_labels,
            y_prob=probabilities,
            n_bins=config_manager.n_bins_calibraion_curve
        )
    }

    # Validate and filter metrics to compute
    if metrics_to_compute is None:
        metrics_to_compute = available_metrics.keys()  # Compute all if not specified
    else:
        invalid_metrics = [m for m in metrics_to_compute if m not in available_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metric(s) requested: {invalid_metrics}. "
                             f"Available options: {list(available_metrics.keys())}")

    # Convert inputs to numpy arrays and ensure integer type for labels
    probabilities = np.array(probabilities)
    predictions = np.array(predictions).astype(int)
    true_labels = np.array(true_labels).astype(int)

    # Initialize calibration metric calculators
    ece_calculator = CalibrationMetricsEnum.get_metric_class('ECE')(
        num_bins=config_manager.n_bins
    )
    mce_calculator = CalibrationMetricsEnum.get_metric_class('MCE')(
        num_bins=config_manager.n_bins
    )

    # Confidence-based ECE calculators
    conf_thresholds = config_manager.conf_thresholds_list
    conf_ece_calculators = {
        threshold: CalibrationMetricsEnum.get_metric_class('ConfECE')(
            num_bins=config_manager.n_bins,
            confidence_threshold=threshold
        )
        for threshold in conf_thresholds
    }

    # Compute requested metrics
    computed_metrics = {metric: available_metrics[metric]() for metric in metrics_to_compute}

    # If 'calibration_curve' is computed, unpack tuple values
    if 'calibration_curve' in computed_metrics:
        mean_predicted_probs, true_probs, bin_counts = computed_metrics['calibration_curve']
        computed_metrics.update({
            'calibration_curve_mean_predicted_probs': mean_predicted_probs,
            'calibration_curve_true_probs': true_probs,
            'calibration_curve_bin_counts': bin_counts
        })
        del computed_metrics['calibration_curve']  # Remove tuple key

    return computed_metrics
