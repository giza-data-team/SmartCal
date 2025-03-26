import logging
import numpy as np
import pandas as pd
import os
import random
from sqlalchemy.orm import Session
from sklearn.model_selection import StratifiedKFold
from config.configuration_manager.configuration_manager import ConfigurationManager
from config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from utils.calibrators_hyperparameters import get_all_calibrator_combinations
from utils.queries import fetch_completed_datasets, extract_uncalibrated_data
from utils.cal_metrics import compute_calibration_metrics

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config_manager = ConfigurationManager()

def get_n_random_calibration_combinations(combinations_dicts, n, seed=config_manager.random_seed):
    if seed is not None:
        random.seed(seed)
    n = min(n, len(combinations_dicts))
    return random.sample(combinations_dicts, n)

def apply_calibration_with_cv(calibrator_dict, uncalibrated_probs_valid, y_valid, uncalibrated_probs_test, n_splits=config_manager.cal_tune_kfolds):
    algo = calibrator_dict.get("Calibration_Algorithm")
    if isinstance(algo, str):
        try:
            algo = CalibrationAlgorithmTypesEnum[algo]
        except KeyError:
            logger.error(f"Invalid Calibration Algorithm: {algo}")
            raise ValueError(f"Invalid Calibration Algorithm: {algo}")
    
    hyperparams = {k: v for k, v in calibrator_dict.items() if k != "Calibration_Algorithm"}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config_manager.random_seed)
    calibrated_probs_valid = np.zeros_like(uncalibrated_probs_valid)
    
    for train_idx, val_idx in skf.split(uncalibrated_probs_valid, y_valid):
        X_train, X_val = uncalibrated_probs_valid[train_idx], uncalibrated_probs_valid[val_idx]
        y_train, y_val = y_valid[train_idx], y_valid[val_idx]
        
        calibrator = algo(**hyperparams)
        calibrator.fit(X_train, y_train)
        calibrated_probs_valid[val_idx] = calibrator.predict(X_val)
    
    final_calibrator = algo(**hyperparams)
    final_calibrator.fit(uncalibrated_probs_valid, y_valid)
    calibrated_probs_test = final_calibrator.predict(uncalibrated_probs_test)
    
    return calibrated_probs_test, calibrated_probs_valid

def save_to_csv_incrementally(result, path):
    """
    Saves a single row of calibration results to a CSV file incrementally.
    
    Args:
        - result (dict): A single row of calibration results.
        - path (str): File path to save CSV.
    """
    df = pd.DataFrame([result])

    # Append new result to CSV, creating the file if it doesn’t exist
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    logger.info(f"Saved result for {result['Dataset Name']} - {result['Classification Model']} (n={result['n']})")


def process_random_baseline(db: Session, table, n_list, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    completed_datasets = fetch_completed_datasets(db, table)
    all_combinations = get_all_calibrator_combinations()
    
    for dataset_name in completed_datasets:
        dataset_rows = extract_uncalibrated_data(db, table, dataset_name)
        if not dataset_rows:
            continue
        
        for row in dataset_rows:
            uncalibrated_probs_valid = row["uncalibrated_probs_cal_set"]
            uncalibrated_probs_test = row["uncalibrated_probs_test_set"]
            y_cal = row["ground_truth_cal_set"]
            y_test = row["ground_truth_test_set"]
            
            best_configuration = None
            best_ece = float("inf")
            best_metrics = None
            
            for n in n_list:
                random_combinations = get_n_random_calibration_combinations(all_combinations, n)
                
                for calibrator_dict in random_combinations:
                    try:
                        calibrated_probs_test, calibrated_probs_valid = apply_calibration_with_cv(
                            calibrator_dict, uncalibrated_probs_valid, y_cal, uncalibrated_probs_test
                        )
                        
                        metrics = compute_calibration_metrics(calibrated_probs_valid, np.argmax(calibrated_probs_valid, axis=1), y_cal)
                        
                        if metrics["ECE"] < best_ece:
                            best_ece = metrics["ECE"]
                            best_configuration = calibrator_dict
                            best_metrics = metrics
                    
                    except Exception as e:
                        logger.error(f"Error applying calibration: {e}")
            
            if best_configuration:
                calibrated_probs_test, _ = apply_calibration_with_cv(
                    best_configuration, uncalibrated_probs_valid, y_cal, uncalibrated_probs_test
                )
                test_metrics = compute_calibration_metrics(calibrated_probs_test, np.argmax(calibrated_probs_test, axis=1), y_test)
                
                best_result = {
                    "Dataset Name": dataset_name,
                    "Classification Model": row["classification_model"],
                    "Best Calibrator": best_configuration["Calibration_Algorithm"],
                    "Hyperparameters": str({k: v for k, v in best_configuration.items() if k != "Calibration_Algorithm"}),
                    "ECE (Cal)": best_metrics["ECE"],
                    "ECE (Test)": test_metrics["ECE"],
                    "LogLoss (Cal)": best_metrics.get("LogLoss", None),
                    "LogLoss (Test)": test_metrics.get("LogLoss", None),
                    "F1-score (Cal)": best_metrics.get("F1-score", None),
                    "F1-score (Test)": test_metrics.get("F1-score", None)
                }
                save_to_csv_incrementally(best_result, output_path)
