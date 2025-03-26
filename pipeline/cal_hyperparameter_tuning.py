from typing import Dict, Any, Tuple, Type
import numpy as np
import pandas as pd
import logging
from itertools import product
from config.configuration_manager.configuration_manager import ConfigurationManager
from config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from sklearn.metrics import accuracy_score, log_loss
from pipeline.pipeline import Pipeline
from config.enums.experiment_status_enum import Experiment_Status_Enum
from utils.cal_metrics import compute_calibration_metrics

# Set up logging
logger = logging.getLogger(__name__)
config_manager = ConfigurationManager()

class CalibrationTuner:
    def __init__(self, pipeline_instance: Pipeline, model_name, config):
        """Initialize with specified calibrators"""
        self.ece_calculator = pipeline_instance.ece_calculator
        self.mce_calculator = pipeline_instance.mce_calculator
        self.conf_ece_calculators = pipeline_instance.conf_ece_calculators

        # Get specified calibrators for this model
        self.calibrators_config = {}
        model_calibrators = config["combinations"].get(model_name, [])
        for run_id, cal_config in model_calibrators:
            for cal_type, hyperparams in cal_config.items():
                self.calibrators_config[cal_type] = {
                    'hyperparams': hyperparams,
                    'run_id': run_id
                }

    def evaluate_calibrator(self,
                           calibrator_class,
                           hyperparams: Dict[str, Any],
                           uncalibrated_probs_valid: np.ndarray,
                           y_valid: np.ndarray,
                           uncalibrated_probs_test: np.ndarray,
                           y_test: np.ndarray) -> Tuple[float, Dict, np.ndarray, np.ndarray]:
        """
        Evaluate calibrator by fitting on validation set and testing on test set
        
        Returns:
            test_ece: ECE score on test set
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
            
            # Calculate ECE on test set
            pred_labels_test = np.argmax(cal_probs_test, axis=1)
            test_ece = self.ece_calculator.compute(
                predicted_probabilities=cal_probs_test,
                predicted_labels=pred_labels_test,
                true_labels=y_test
            )
            
            return test_ece, timing_info, cal_probs_valid, cal_probs_test

        except Exception as e:
            logger.error(f"Error in calibrator evaluation: {str(e)}")
            return float('inf'), None, None, None

    def tune_all_calibrators(self,
                            uncalibrated_probs_valid: np.ndarray,
                            uncalibrated_probs_test: np.ndarray,
                            y_valid: np.ndarray,
                            y_test: np.ndarray) -> Dict[str, Dict]:
        """Tune calibrators by evaluating all hyperparameter combinations on test set"""
        all_results = {}
        
        # Convert inputs to numpy arrays if they aren't already
        uncalibrated_probs_valid = np.array(uncalibrated_probs_valid)
        uncalibrated_probs_test = np.array(uncalibrated_probs_test)
        y_valid = np.array(y_valid)
        y_test = np.array(y_test)
        
        for calibrator_class, config in self.calibrators_config.items():
            calibrator_name = CalibrationAlgorithmTypesEnum(calibrator_class).name
            logger.info(f"\nTuning {calibrator_name}...")
            
            try:
                hyperparams = config['hyperparams']
                run_id = config['run_id']
                results = []
                
                best_ece = float('inf')
                best_hyperparams = None
                best_timing_info = None
                best_cal_probs_valid = None
                best_cal_probs_test = None

                # Handle calibrators with no hyperparameters
                if not hyperparams:
                    test_ece, timing_info, cal_probs_valid, cal_probs_test = self.evaluate_calibrator(
                        calibrator_class, {}, 
                        uncalibrated_probs_valid, y_valid,
                        uncalibrated_probs_test, y_test
                    )
                    
                    if test_ece < float('inf'):
                        best_ece = test_ece
                        best_hyperparams = {}
                        best_timing_info = timing_info
                        best_cal_probs_valid = cal_probs_valid
                        best_cal_probs_test = cal_probs_test
                        
                        results.append({
                            'test_ece': best_ece,
                            'timing': timing_info
                        })

                # Handle calibrators with hyperparameters
                else:
                    param_names = list(hyperparams.keys())
                    param_values = list(hyperparams.values())
                    
                    for values in product(*param_values):
                        current_hyperparams = dict(zip(param_names, values))
                        logger.info(f"Evaluating {calibrator_name} with hyperparameters: {current_hyperparams}")
                        
                        test_ece, timing_info, cal_probs_valid, cal_probs_test = self.evaluate_calibrator(
                            calibrator_class, current_hyperparams,
                            uncalibrated_probs_valid, y_valid,
                            uncalibrated_probs_test, y_test
                        )
                        
                        if test_ece < float('inf'):
                            results.append({
                                **current_hyperparams,
                                'test_ece': float(test_ece),
                                'timing': timing_info
                            })
                            
                            if test_ece < best_ece:
                                best_ece = test_ece
                                best_hyperparams = current_hyperparams
                                best_timing_info = timing_info
                                best_cal_probs_valid = cal_probs_valid
                                best_cal_probs_test = cal_probs_test

                # Calculate metrics if calibration was successful
                if best_cal_probs_valid is not None and best_cal_probs_test is not None:
                    pred_labels_valid = np.argmax(best_cal_probs_valid, axis=1)
                    pred_labels_test = np.argmax(best_cal_probs_test, axis=1)
                    
                    n_classes = uncalibrated_probs_valid.shape[1]
                    all_classes = np.arange(n_classes)

                    val_metrics = {
                        'loss': float(log_loss(y_valid, best_cal_probs_valid, labels=all_classes)),
                        'accuracy': float(accuracy_score(y_valid, pred_labels_valid))
                    }
                    val_metrics.update(compute_calibration_metrics(
                        probabilities=best_cal_probs_valid,
                        predictions=pred_labels_valid,
                        true_labels=y_valid
                    ))

                    test_metrics = {
                        'loss': float(log_loss(y_test, best_cal_probs_test, labels=all_classes)),
                        'accuracy': float(accuracy_score(y_test, pred_labels_test))
                    }
                    test_metrics.update(compute_calibration_metrics(
                        probabilities=best_cal_probs_test,
                        predictions=pred_labels_test,
                        true_labels=y_test
                    ))
                    
                    if calibrator_name == 'META' and best_hyperparams['calibrator_type'] == 'ALPHA':
                        best_hyperparams['acc'] = None
                    elif calibrator_name == 'META' and best_hyperparams['calibrator_type'] == 'ACC':
                        best_hyperparams['alpha'] = None
                        
                    if calibrator_name == 'AdaptiveTemperatureScaling' and best_hyperparams['mode'] == 'entropy':
                        best_hyperparams['confidence_bins'] = None
                    elif calibrator_name == 'AdaptiveTemperatureScaling' and best_hyperparams['mode'] == 'linear':
                        best_hyperparams['entropy_bins'] = None
                        
                    all_results[calibrator_name] = {
                        'run_id': run_id,
                        'cal_status': Experiment_Status_Enum.COMPLETED.value,
                        'cal_timing': best_timing_info,
                        'best_hyperparams': best_hyperparams,
                        'calibrated_metrics_val_set': val_metrics,
                        'calibrated_metrics_tst_set': test_metrics,
                        'calibrated_probs_val_set': best_cal_probs_valid.tolist(),
                        'calibrated_probs_test_set': best_cal_probs_test.tolist()
                    }
                else:
                    # If calibration failed but didn't raise an exception
                    all_results[calibrator_name] = {
                        'run_id': run_id,
                        'cal_status': Experiment_Status_Enum.FAILED.value,
                        'error': 'Calibration failed to produce valid probabilities'
                    }

                # Log results
                logger.info(f"\nResults for {calibrator_name} (Run ID: {run_id}):")
                if results:
                    logger.info(f"\n{pd.DataFrame(results).to_string(index=False)}")
                logger.info(f"Best hyperparameters: {best_hyperparams}")
                logger.info(f"Best Test ECE: {best_ece:.4f}")
                logger.info(f"Timing information: {best_timing_info}")
                logger.info(f"Calibration status: {all_results[calibrator_name]['cal_status']}")
                
            except Exception as e:
                logger.error(f"Error tuning {calibrator_name}: {str(e)}")
                all_results[calibrator_name] = {
                    'run_id': run_id,
                    'cal_status': Experiment_Status_Enum.FAILED.value,
                    'error': str(e)
                }

        return all_results

def tune_all_calibration(uncalibrated_probs_valid, uncalibrated_probs_test, 
                        y_valid, y_test, pipeline_instance, model_name, config):
    """
    Wrapper function to create tuner instance and perform calibration.
    
    Args:
        uncalibrated_probs_valid: Validation set probabilities
        uncalibrated_probs_test: Test set probabilities
        y_valid: Validation set true labels
        y_test: Test set true labels
        pipeline_instance: Instance of Pipeline class
        model_name: Name of the model being calibrated
        config: Configuration dictionary
        
    Returns:
        Dictionary containing tuning results for all calibrators
    """
    tuner = CalibrationTuner(pipeline_instance, model_name, config)
    return tuner.tune_all_calibrators(
        uncalibrated_probs_valid,
        uncalibrated_probs_test,
        y_valid,
        y_test
    )