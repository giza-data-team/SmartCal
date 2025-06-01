from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import logging
from itertools import product
from sklearn.metrics import accuracy_score, log_loss

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from pipeline.pipeline import Pipeline
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from smartcal.utils.cal_metrics import compute_calibration_metrics

from smartcal.config.enums.calibration_metrics_enum import CalibrationMetricsEnum
from smartcal.metrics.brier_score import calculate_brier_score
from smartcal.metrics.log_loss import calculate_log_loss
from smartcal.config.enums.calibration_error_code_enum import CalibrationErrorCode
# Set up logging
logger = logging.getLogger(__name__)
config_manager = ConfigurationManager()


class CalibrationTuner:
    def __init__(self, pipeline_instance: Pipeline, model_name, config, use_kfold=False, n_splits=5):
        """Initialize with specified calibrators"""
        self.ece_calculator = pipeline_instance.ece_calculator
        self.mce_calculator = pipeline_instance.mce_calculator
        self.conf_ece_calculators_thres = pipeline_instance.conf_ece_calculators_thres
        self.use_kfold = use_kfold
        self.n_splits = n_splits
        if self.use_kfold:
            from sklearn.model_selection import StratifiedKFold
            self.kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=config_manager.random_seed)

        # Get specified calibrators and metrics for this model
        self.calibrators_config = []
        self.metric_config = {}

        model_configs = config["combinations"].get(model_name, [])
        for run_id, metric, cal_config in model_configs:
            for cal_type, hyperparams in cal_config.items():
                self.calibrators_config.append({
                    'cal_type': cal_type,
                    'hyperparams': hyperparams,
                    'run_id': run_id
                })
                self.metric_config[run_id] = metric

    def compute_metric(self, probabilities, predictions, true_labels, metric_enum):
        """Compute the specified metric"""
        if metric_enum == CalibrationMetricsEnum.ECE:
            return self.ece_calculator.compute(probabilities, predictions, true_labels)
        elif metric_enum == CalibrationMetricsEnum.MCE:
            return self.mce_calculator.compute(probabilities, predictions, true_labels)
        elif metric_enum == CalibrationMetricsEnum.ConfECE:
            return self.conf_ece_calculators_thres.compute(probabilities, predictions, true_labels)
        elif metric_enum == CalibrationMetricsEnum.brier_score:
            return calculate_brier_score(true_labels, probabilities)
        elif metric_enum == CalibrationMetricsEnum.log_loss:
            return calculate_log_loss(true_labels, probabilities)
        else:
            raise ValueError(f"Unknown metric enum: {metric_enum}")

    def evaluate_calibrator_cv(self, calibrator_class, hyperparams, uncalibrated_probs, y, metric_enum):
        """Evaluate calibrator using stratified k-fold cross-validation."""
        metric_scores = []
        total_fit_time = 0
        total_predict_time = 0

        X = np.array(uncalibrated_probs)
        y = np.array(y)
        n_classes = X.shape[1]
        all_classes = np.arange(n_classes)

        try:
            for train_idx, val_idx in self.kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                calibrator = calibrator_class(**hyperparams)
                calibrator.fit(X_train, y_train)
                cal_probs_val = calibrator.predict(X_val)

                timing = calibrator.get_timing()
                total_fit_time += timing['fit']
                total_predict_time += timing['predict']

                pred_labels = np.argmax(cal_probs_val, axis=1)
                metric_value = self.compute_metric(cal_probs_val, pred_labels, y_val, metric_enum)
                metric_scores.append(metric_value)

            avg_metric = np.mean(metric_scores)
            timing_info = {
                'fit': total_fit_time / self.n_splits,
                'predict': total_predict_time / self.n_splits
            }
            return avg_metric, timing_info
        except Exception as e:
            logger.error(f"CV evaluation error: {str(e)}")
            return float('inf'), None

    def evaluate_calibrator(self,
                            calibrator_class,
                            hyperparams: Dict[str, Any],
                            uncalibrated_probs_valid: np.ndarray,
                            y_valid: np.ndarray,
                            uncalibrated_probs_test: np.ndarray,
                            y_test: np.ndarray,
                            metric_enum: CalibrationMetricsEnum) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
        """
        Evaluate calibrator by fitting on validation set and testing on test set

        Returns:
            test_metric: Metric score on test set
            timing_info: Dictionary with timing information
            cal_probs_valid: Calibrated probabilities for validation set
            cal_probs_test: Calibrated probabilities for test set
        """
        # Convert to numpy arrays if not already
        X_valid = np.array(uncalibrated_probs_valid)
        y_valid = np.array(y_valid)
        X_test = np.array(uncalibrated_probs_test)
        y_test = np.array(y_test)

        try:
            # Initialize and fit calibrator on validation set
            calibrator = calibrator_class(**hyperparams)
            calibrator.fit(X_valid, y_valid)

            # Predict on both validation and test sets
            cal_probs_valid = calibrator.predict(X_valid)
            cal_probs_test = calibrator.predict(X_test)

            # Get timing information
            timing_info = calibrator.get_timing()

            # Calculate specified metric on test set
            pred_labels_test = np.argmax(cal_probs_test, axis=1)
            test_metric = self.compute_metric(
                probabilities=cal_probs_test,
                predictions=pred_labels_test,
                true_labels=y_test,
                metric_enum=metric_enum
            )

            return test_metric, timing_info, cal_probs_valid, cal_probs_test

        except Exception as e:
            logger.error(f"Error in calibrator evaluation: {str(e)}")
            return float('inf'), None, None, None

    def tune_all_calibrators(self,
                             uncalibrated_probs_valid: np.ndarray,
                             uncalibrated_probs_test: np.ndarray,
                             y_valid: np.ndarray,
                             y_test: np.ndarray) -> Dict[str, Dict]:
        """Tune calibrators by evaluating all hyperparameter combinations once and computing all metrics"""
        all_results = {}

        # Convert inputs to numpy arrays
        uncalibrated_probs_valid = np.array(uncalibrated_probs_valid)
        uncalibrated_probs_test = np.array(uncalibrated_probs_test)
        y_valid = np.array(y_valid)
        y_test = np.array(y_test)

        # Group configurations by calibrator type
        calibrator_groups = {}
        for config in self.calibrators_config:
            cal_type = config['cal_type']
            if cal_type not in calibrator_groups:
                calibrator_groups[cal_type] = []
            calibrator_groups[cal_type].append(config)

        for calibrator_class, configs in calibrator_groups.items():
            try:
                calibrator_name = CalibrationAlgorithmTypesEnum(calibrator_class).name

                # Initialize results storage for all run_ids
                results_storage = {
                    config['run_id']: {
                        'metric': float('inf'),
                        'hyperparams': None,
                        'timing_info': None,
                        'cal_probs_valid': None,
                        'cal_probs_test': None
                    } for config in configs
                }

                logger.info(f"\nTuning {calibrator_name}...")

                # Get hyperparameter grid based on calibrator type
                hyperparams_config = configs[0]['hyperparams']  # All configs for same calibrator have same hyperparams

                if not hyperparams_config:  # For calibrators with no hyperparameters (e.g., Isotonic)
                    param_grid = [{}]
                else:
                    # Generate parameter grid based on calibrator type
                    param_names = list(hyperparams_config.keys())
                    param_values = list(hyperparams_config.values())
                    param_grid = [dict(zip(param_names, v)) for v in product(*param_values)]

                # Evaluate each hyperparameter combination
                for current_hyperparams in param_grid:
                    logger.info(f"Evaluating {calibrator_name} with hyperparameters: {current_hyperparams}")

                    try:
                        # Fit calibrator once for this hyperparameter combination
                        calibrator = calibrator_class(**current_hyperparams)
                        calibrator.fit(uncalibrated_probs_valid, y_valid)

                        # Try predicting
                        cal_probs_valid = calibrator.predict(uncalibrated_probs_valid)
                        cal_probs_test = calibrator.predict(uncalibrated_probs_test)
                        timing_info = calibrator.get_timing()

                        # Calculate predictions once
                        pred_labels_valid = np.argmax(cal_probs_valid, axis=1)
                        pred_labels_test = np.argmax(cal_probs_test, axis=1)

                    except ValueError as e:
                        if isinstance(e.args[0], tuple):
                            error_code = e.args[0][0]
                            if error_code in (CalibrationErrorCode.NANS_DETECTED, CalibrationErrorCode.ALMOST_CONSTANT_PROBS):
                                logger.warning(f"Skipping {calibrator_name} with hyperparameters {current_hyperparams} due to NaNs or bad parameters during prediction: {e}")
                                continue
                        # Unknown or critical ValueError: re-raise
                        raise

                    # Evaluate all metrics for this calibration
                    for config in configs:
                        run_id = config['run_id']
                        metric_enum = self.metric_config[run_id]

                        # Compute metric for test set
                        test_metric = self.compute_metric(
                            probabilities=cal_probs_test,
                            predictions=pred_labels_test,
                            true_labels=y_test,
                            metric_enum=metric_enum
                        )

                        # updating the results for the best metric
                        if test_metric < results_storage[run_id]['metric']:
                            # Calculate all other metrics for this configuration
                            all_metrics = {}
                            for metric_name, metric_enum_value in CalibrationMetricsEnum.__members__.items():
                                calculated_metric = self.compute_metric(
                                    probabilities=cal_probs_test,
                                    predictions=pred_labels_test,
                                    true_labels=y_test,
                                    metric_enum=metric_enum_value
                                )
                                all_metrics[metric_name] = calculated_metric

                            # Update with all metrics and the configuration
                            results_storage[run_id].update({
                                'metric': test_metric,
                                'all_metrics': all_metrics,
                                'hyperparams': current_hyperparams,
                                'timing_info': timing_info,
                                'cal_probs_valid': cal_probs_valid,
                                'cal_probs_test': cal_probs_test
                            })

                # Create final results for each run_id using their best configurations
                for config in configs:
                    run_id = config['run_id']
                    metric_enum = self.metric_config[run_id]
                    best_result = results_storage[run_id]

                    if best_result['cal_probs_valid'] is not None:
                        pred_labels_valid = np.argmax(best_result['cal_probs_valid'], axis=1)
                        pred_labels_test = np.argmax(best_result['cal_probs_test'], axis=1)

                        n_classes = uncalibrated_probs_valid.shape[1]
                        all_classes = np.arange(n_classes)

                        # Calculate metrics for validation and test sets
                        val_metrics = {
                            'loss': float(log_loss(y_valid, best_result['cal_probs_valid'], labels=all_classes)),
                            'accuracy': float(accuracy_score(y_valid, pred_labels_valid))
                        }
                        val_metrics.update(compute_calibration_metrics(
                            probabilities=best_result['cal_probs_valid'],
                            predictions=pred_labels_valid,
                            true_labels=y_valid
                        ))

                        test_metrics = {
                            'loss': float(log_loss(y_test, best_result['cal_probs_test'], labels=all_classes)),
                            'accuracy': float(accuracy_score(y_test, pred_labels_test))
                        }
                        test_metrics.update(compute_calibration_metrics(
                            probabilities=best_result['cal_probs_test'],
                            predictions=pred_labels_test,
                            true_labels=y_test
                        ))

                        if calibrator_name == 'META' and best_result['hyperparams']['calibrator_type'] == 'ALPHA':
                            best_result['hyperparams']['acc'] = None
                        elif calibrator_name == 'META' and best_result['hyperparams']['calibrator_type'] == 'ACC':
                            best_result['hyperparams']['alpha'] = None

                        if calibrator_name == 'AdaptiveTemperatureScaling' and best_result['hyperparams'][
                            'mode'] == 'entropy':
                            best_result['hyperparams']['confidence_bins'] = None
                        elif calibrator_name == 'AdaptiveTemperatureScaling' and best_result['hyperparams'][
                            'mode'] == 'linear':
                            best_result['hyperparams']['entropy_bins'] = None

                        # Store final results
                        all_results[f"{calibrator_name}_{run_id}"] = {
                            'run_id': run_id,
                            'calibration_metric': metric_enum.name,
                            'cal_status': Experiment_Status_Enum.COMPLETED.value,
                            'cal_timing': best_result['timing_info'],
                            'best_hyperparams': best_result['hyperparams'],
                            'calibrated_metrics_val_set': val_metrics,
                            'calibrated_metrics_tst_set': test_metrics,
                            'all_metrics': best_result['all_metrics'],  # Include all metrics
                            'calibrated_probs_val_set': best_result['cal_probs_valid'].tolist(),
                            'calibrated_probs_test_set': best_result['cal_probs_test'].tolist()
                        }
                    else:
                        all_results[f"{calibrator_name}_{run_id}"] = {
                            'run_id': run_id,
                            'calibration_metric': metric_enum.name,
                            'cal_status': Experiment_Status_Enum.FAILED.value,
                            'error': 'Calibration failed to produce valid probabilities'
                        }

            except Exception as e:
                logger.error(f"Error tuning {calibrator_name}: {str(e)}")
                for config in configs:
                    run_id = config['run_id']
                    metric_enum = self.metric_config[run_id]
                    all_results[f"{calibrator_name}_{run_id}"] = {
                        'run_id': run_id,
                        'calibration_metric': metric_enum.name,
                        'cal_status': Experiment_Status_Enum.FAILED.value,
                        'error': str(e)
                    }

        return all_results


def tune_all_calibration(uncalibrated_probs_valid, uncalibrated_probs_test,
                         y_valid, y_test, pipeline_instance, model_name, config, use_kfold=False):
    """
    Wrapper function to create tuner instance and perform calibration.
    """
    tuner = CalibrationTuner(pipeline_instance, model_name, config, use_kfold)
    return tuner.tune_all_calibrators(
        uncalibrated_probs_valid,
        uncalibrated_probs_test,
        y_valid,
        y_test
    )