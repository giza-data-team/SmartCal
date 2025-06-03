from typing import Dict, Callable, Union, List, Any
import numpy as np
import logging
from bayes_opt import BayesianOptimization
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold
import time

from smartcal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from smartcal.utils.cal_metrics import compute_calibration_metrics
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.bayesian_optimization.bayesian_hyperparameters import param_mappings, param_spaces
from smartcal.meta_features_extraction.meta_features_extraction import MetaFeaturesExtractor

config_manager = ConfigurationManager()


def get_metric_value(metrics: Dict[str, Any], y_true: np.array, y_pred: np.array, metric_name: str) -> float:
    """
    Get a specific metric value from computed metrics or calculate it if needed.
    
    Args:
        metrics: Dictionary of pre-computed calibration metrics
        y_true: True labels
        y_pred: Predicted probabilities
        metric_name: Name of the metric to retrieve
        
    Returns:
        float: The value of the requested metric
    """
    metric_evaluators = {
        'ECE': lambda m, y_t, y_p: m['ece'],
        'MCE': lambda m, y_t, y_p: m['mce'],
        'brier_score': lambda m, y_t, y_p: m['brier_score'],
        'ConfECE': lambda m, y_t, y_p: np.mean(m['conf_ece']),
        'log_loss': lambda m, y_t, y_p: log_loss(y_t, y_p)
    }
    
    if metric_name not in metric_evaluators:
        raise ValueError(f"Unsupported metric: {metric_name}")
    
    return metric_evaluators[metric_name](metrics, y_true, y_pred)


class CalibrationOptimizer:
    def __init__(self, meta_model, k_folds=config_manager.cal_tune_kfolds):
        """
        Initialize the CalibrationOptimizer with a meta-model and cross-validation settings.

        Args:
            meta_model: A model capable of recommending calibration algorithms
            k_folds (int, optional): Number of cross-validation folds.
        """
        self.meta_model = meta_model
        self.k_folds = k_folds

    def get_calibrator_recommendations(self, y_true, y_pred):
        """
        Gets the calibrator recommendations based on predicted probabilities and true labels.

        This function extracts meta-features from the given calibration data and then
        uses the meta-model to predict which calibrators should be used, and their associated confidence scores.

        Args:
            y_true (np.array): True labels of the calibration dataset.
            y_pred (np.array): Predicted probabilities from a classifier.

        Returns:
            list: A list of tuples, each containing a calibrator name and its corresponding normalized confidence score.
        """
        extractor = MetaFeaturesExtractor()
        meta_features = extractor.process_features(y_true, y_pred)
        recommendations = self.meta_model.predict_best_model(meta_features)
        
        # Normalize confidence scores to sum to 1
        calibrators, scores = zip(*recommendations)
        normalized_scores = np.array(scores) / sum(scores)
        
        logging.info(f"Recommendations: {recommendations}")
        logging.info(f"Normalized Scores: {normalized_scores}")
        
        return list(zip(calibrators, normalized_scores))

    def allocate_iterations(self, normalized_scores, total_iterations):
        """
        Allocates the total number of iterations to each calibrator based on its normalized confidence score.

        This ensures that the sum of all allocations equals the total_iterations and
        tries to allocate iterations proportional to confidence.

        Args:
            normalized_scores (list): List of normalized confidence scores for each calibrator.
            total_iterations (int): Total number of iterations to allocate.

        Returns:
            np.array: The final allocation of iterations for each calibrator.
        """

        logging.info(f"Normalized scores: {normalized_scores}")

        # Ensure normalized_scores is a numpy array
        normalized_scores = np.array(normalized_scores)

        # Calculate raw allocations
        raw_allocations = normalized_scores * total_iterations
        logging.info(f"Raw allocations: {raw_allocations}")

        # Floor the raw allocations
        allocations = np.floor(raw_allocations).astype(int)
        logging.info(f"After floor: {allocations}")

        # Calculate remainders
        remainders = raw_allocations - allocations
        logging.info(f"Remainders: {remainders}")

        # Calculate remaining iterations
        remaining_iterations = total_iterations - np.sum(allocations)
        logging.info(f"Remaining iterations: {remaining_iterations}")

        # Allocate remaining iterations to the calibrators with the largest remainders
        indices = np.argsort(-remainders)[:remaining_iterations]
        logging.info(f"Indices for remaining: {indices}")

        allocations[indices] += 1
        logging.info(f"Final allocations: {allocations}")

        return allocations

    def optimize_calibrator(self, calibrator_name: str, X_cal, y_cal, n_iterations: int, metric: str = config_manager.metric) -> Dict:
        """
        Optimize hyperparameters for a specific calibration algorithm using Bayesian Optimization.

        This method handles mixed parameter types (continuous, categorical, integer)
        and uses cross-validation to evaluate calibrator performance.

        Args:
            calibrator_name (str): Name of the calibration algorithm
            X_cal (array-like): Calibration set features
            y_cal (array-like): Calibration set labels
            n_iterations (int): Number of Bayesian optimization iterations

        Returns:
            Dict: Optimization results including best parameters, score, and metrics
        """

        calibrator_class = CalibrationAlgorithmTypesEnum[calibrator_name]

        bayesian_opt_start_time = time.perf_counter()

        def objective(**params):
            # Convert parameters back to their original types
            converted_params = {}
            mapping = param_mappings.get(calibrator_name, {})

            for param_name, param_value in params.items():
                if param_name in mapping:
                    # Categorical parameter - map back to string
                    choices = list(mapping[param_name].keys())
                    idx = int(round(param_value))
                    converted_params[param_name] = choices[idx % len(choices)]
                elif param_name.endswith('_iters') or param_name.endswith('_iter') or 'max_iter' in param_name:
                    # Integer parameter
                    converted_params[param_name] = int(round(param_value))
                elif param_name == 'n_bins' or param_name == 'num_bins' or 'bins' in param_name:
                    # Integer parameter for bins
                    converted_params[param_name] = int(round(param_value))
                else:
                    # Continuous parameter
                    converted_params[param_name] = param_value

            calibrator = calibrator_class(**converted_params)
            scores = []

            kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=config_manager.random_seed)
            for train_idx, val_idx in kf.split(X_cal, y_cal):
                train_idx = np.array(train_idx, dtype=int)
                val_idx = np.array(val_idx, dtype=int)

                X_train, X_val = X_cal[train_idx], X_cal[val_idx]
                y_train, y_val = y_cal[train_idx], y_cal[val_idx]

                calibrator.fit(X_train, y_train)
                calibrated_probs = calibrator.predict(X_val)
                calibrated_predicted_label = np.argmax(calibrated_probs, axis=1).tolist()
                metrics = compute_calibration_metrics(calibrated_probs, calibrated_predicted_label, y_val)
                
                # Get the metric value using the centralized function
                metric_value = get_metric_value(metrics, y_val, calibrated_probs, metric)
                scores.append(metric_value)

            return -np.mean(scores)

        # Get the parameter space for this calibrator
        pbounds = param_spaces.get(calibrator_name, {})

        if not pbounds:
            # No parameters to optimize
            calibrator = calibrator_class()
            calibrator.fit(X_cal, y_cal)
            calibrated_probs = calibrator.predict(X_cal)
            calibrated_predicted_label = np.argmax(calibrated_probs, axis=1).tolist()
            metrics = compute_calibration_metrics(calibrated_probs, calibrated_predicted_label, y_cal)

            bayesian_opt_end_time = time.perf_counter()
            bayesian_opt_time = bayesian_opt_end_time - bayesian_opt_start_time

            return {
                'best_params': {},
                'best_score': metrics['ece'],
                'full_metrics': metrics,
                'bayesian_optimization_time': bayesian_opt_time
            }

        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=config_manager.random_seed
        )

        init_points = min(3, n_iterations)
        n_iter = max(0, n_iterations - init_points)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        bayesian_opt_end_time = time.perf_counter()
        bayesian_opt_time = bayesian_opt_end_time - bayesian_opt_start_time

        # Convert best parameters back to original types
        best_params = {}
        mapping = param_mappings.get(calibrator_name, {})

        for param_name, param_value in optimizer.max['params'].items():
            if param_name in mapping:
                choices = list(mapping[param_name].keys())
                idx = int(round(param_value))
                best_params[param_name] = choices[idx % len(choices)]
            elif param_name.endswith('_iters') or param_name.endswith('_iter') or 'max_iter' in param_name:
                best_params[param_name] = int(round(param_value))
            elif param_name == 'n_bins' or param_name == 'num_bins' or 'bins' in param_name:
                best_params[param_name] = int(round(param_value))
            else:
                best_params[param_name] = param_value

        # Fit with the best parameters to get full metrics
        best_calibrator = calibrator_class(**best_params)
        best_calibrator.fit(X_cal, y_cal)
        calibrated_probs = best_calibrator.predict(X_cal)
        calibrated_predicted_label = np.argmax(calibrated_probs, axis=1).tolist()
        full_metrics = compute_calibration_metrics(calibrated_probs, calibrated_predicted_label, y_cal)

        return {
            'best_params': best_params,
            'best_score': -optimizer.max['target'],
            'full_metrics': full_metrics,
            'bayesian_optimization_time': bayesian_opt_time
        }

    def run_optimization(self, dataset_results: Dict, total_iterations: int = 30, metric: str = config_manager.metric):
        """Run the complete optimization process, saving only the best calibrator's results"""
        results = {}

        try:
            # Ensure X_cal and y_cal are NumPy arrays
            X_cal = np.array(dataset_results['calibration_set_predicted_probabilities'])
            y_cal = np.array(dataset_results['calibration_set_true_labels'])

            X_test = np.array(dataset_results['test_set_predicted_probabilities'])
            y_test = np.array(dataset_results['test_set_true_labels'])

            # Validate shapes
            if X_cal.ndim != 2 or X_test.ndim != 2:
                raise ValueError("Input data must be 2D arrays (samples x features).")

            # Get recommendations and allocate iterations
            recommendations = self.get_calibrator_recommendations(y_cal, X_cal)
            normalized_scores = [score for _, score in recommendations]
            iteration_allocations = self.allocate_iterations(normalized_scores, total_iterations)

            best_calibrator = None
            best_metric_value = float('inf')
            best_hyperparameters = "None"

            # Store the metric being optimized
            results["optimization_metric"] = metric

            # Optimize each recommended calibrator
            for (calibrator_name, confidence), n_iterations in zip(recommendations, iteration_allocations):
                if n_iterations > 0:
                    try:
                        optimization_result = self.optimize_calibrator(
                            calibrator_name, X_cal, y_cal, n_iterations, metric
                        )

                        # Skip if optimization failed
                        if optimization_result is None or optimization_result.get('full_metrics') is None:
                            continue

                        # Instantiate and fit the calibrator with best parameters
                        current_params = optimization_result['best_params'].copy()

                        # For params saving purposes
                        param_str = ", ".join(
                            [f"{k}={v}" for k, v in current_params.items()]) if current_params else "None"

                        try:
                            calibrator = CalibrationAlgorithmTypesEnum[calibrator_name](**current_params)
                            calibrator.fit(X_cal, y_cal)

                            # Evaluate on calibration set
                            cal_probs = calibrator.predict(X_cal)
                            cal_predicted_label = np.argmax(cal_probs, axis=1).tolist()
                            cal_metrics = compute_calibration_metrics(cal_probs, cal_predicted_label, y_cal)
                            
                            # Get the specific metric using the centralized function
                            current_metric_value = get_metric_value(cal_metrics, y_cal, cal_probs, metric)

                            # For all our metrics, lower is better
                            if current_metric_value < best_metric_value:
                                best_metric_value = current_metric_value
                                best_calibrator = calibrator
                                best_hyperparameters = param_str

                                # Store metrics
                                results["cal_ece"] = float(cal_metrics['ece'])
                                results["cal_mce"] = float(cal_metrics['mce'])
                                results["cal_conf_ece"] = list(cal_metrics['conf_ece'])
                                results["cal_brier_score"] = float(cal_metrics['brier_score'])
                                results["cal_log_loss"] = float(log_loss(y_cal, cal_probs))
                                results["cal_f1_score_macro"] = float(
                                    f1_score(y_cal, cal_probs.argmax(axis=1), average='macro'))
                                results["cal_f1_score_micro"] = float(
                                    f1_score(y_cal, cal_probs.argmax(axis=1), average='micro'))
                                results["cal_calibration_curve_mean_predicted_probs"] = cal_metrics[
                                    'calibration_curve_mean_predicted_probs']
                                results["cal_calibration_curve_true_probs"] = cal_metrics[
                                    'calibration_curve_true_probs']
                                results["cal_calibration_curve_bin_counts"] = cal_metrics[
                                    'calibration_curve_bin_counts']

                                # Store the optimized metric value
                                results[f"best_{metric}_value"] = float(current_metric_value)

                                results["confidence_score"] = float(confidence)
                                results["allocated_iterations"] = int(n_iterations)
                                results["bayesian_optimization_time"] = optimization_result.get(
                                    'bayesian_optimization_time', None)

                        except Exception as e:
                            logging.warning(f"Failed to evaluate {calibrator_name}: {str(e)}")
                            continue

                    except Exception as e:
                        logging.warning(f"Optimization failed for {calibrator_name}: {str(e)}")
                        continue

            # Store best calibrator info
            results["best_calibrator"] = best_calibrator.__class__.__name__ if best_calibrator else "None"
            results["hyperparameters"] = best_hyperparameters

            # Evaluate best calibrator on test set if available
            if best_calibrator is not None:
                try:
                    test_probs = best_calibrator.predict(X_test)
                    test_predicted_label = np.argmax(test_probs, axis=1).tolist()
                    test_metrics_full = compute_calibration_metrics(test_probs, test_predicted_label, y_test)

                    # Store test metrics
                    results["test_ece"] = float(test_metrics_full['ece'])
                    results["test_mce"] = float(test_metrics_full['mce'])
                    results["test_conf_ece"] = list(test_metrics_full['conf_ece'])
                    results["test_brier_score"] = float(test_metrics_full['brier_score'])
                    results["test_log_loss"] = float(log_loss(y_test, test_probs))
                    results["test_f1_score_macro"] = float(f1_score(y_test, test_probs.argmax(axis=1), average='macro'))
                    results["test_f1_score_micro"] = float(f1_score(y_test, test_probs.argmax(axis=1), average='micro'))

                    # Store calibration curve data
                    results["test_calibration_curve_mean_predicted_probs"] = test_metrics_full[
                        'calibration_curve_mean_predicted_probs']
                    results["test_calibration_curve_true_probs"] = test_metrics_full['calibration_curve_true_probs']
                    results["test_calibration_curve_bin_counts"] = test_metrics_full['calibration_curve_bin_counts']

                except Exception as e:
                    logging.warning(f"Test evaluation failed: {str(e)}")
                    results["test_error"] = str(e)

        except Exception as e:
            logging.error(f"Critical error in run_optimization: {str(e)}")
            results["error"] = str(e)

        return results