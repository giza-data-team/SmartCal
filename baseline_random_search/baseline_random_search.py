import warnings
warnings.filterwarnings("ignore", message="The y_pred values do not sum to one. Make sure to pass probabilities.")

import logging
import numpy as np
import pandas as pd
import os
import random
import json
import ast
import csv
import hashlib
from sqlalchemy.orm import Session
from sklearn.model_selection import StratifiedKFold
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from Package.src.SmartCal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from Package.src.SmartCal.utils.calibrators_hyperparameters import get_all_calibrator_combinations
from Package.src.SmartCal.utils.cal_metrics import compute_calibration_metrics
from Package.src.SmartCal.utils.classification_metrics import compute_classification_metrics
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
import logging
# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

config_manager = ConfigurationManager()

def make_hashable(obj):
    if isinstance(obj, dict):
        return frozenset((k, make_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, set):
        return frozenset(make_hashable(x) for x in obj)
    else:
        return obj

def get_n_random_calibration_combinations(combinations_dicts, n=50, seed=31):
    rng = random.Random(seed)  # local RNG instance
    n = min(n, len(combinations_dicts))
    # Create a list of all combinations with their random values for sorting
    all_combinations = [(rng.random(), combination) for combination in combinations_dicts]
    # Sort the list by the random value (deterministic based on seed)
    all_combinations.sort(key=lambda x: x[0])
    # Take the first n elements and return just the combinations (without the random values)
    return [combination for _, combination in all_combinations[:n]]

def apply_calibration_with_cv(calibrator_dict, uncalibrated_probs_valid, y_valid, uncalibrated_probs_test, n_splits=2): # config_manager.cal_tune_kfolds
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

def flatten(value):
    """
    Flatten a single-element tuple or list to its scalar value.
    """
    if isinstance(value, (tuple, list)) and len(value) == 1:
        return value[0]
    return value

def parse_string(value):
    """
    Attempt to parse a string that looks like a list or tuple.
    """
    try:
        parsed = ast.literal_eval(value)
        # If parsing was successful and it’s a list or tuple, convert to a proper list
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
    except (SyntaxError, ValueError):
        pass
    return value

def convert_tuple_to_list_string(value):
    """
    Convert tuple-like strings to list-like strings.
    """
    if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
        # Replace parentheses with square brackets
        return "[" + value[1:-1] + "]"
    return value

def clean_value(value):
    """
    Clean a value by:
    - Flattening tuple-like structures.
    - Parsing string representations of lists or tuples.
    - Converting tuple-like strings to list-like strings.
    """
    # Flatten single-element tuples or lists
    value = flatten(value)
    
    # If it's a string that looks like a list or tuple, parse it
    if isinstance(value, str):
        value = parse_string(value)

    # If the value is a list after parsing, convert to JSON-style list format
    if isinstance(value, list):
        value = "[" + ", ".join(map(str, value)) + "]"
    
    # Convert any remaining tuple-like strings to list-like strings
    value = convert_tuple_to_list_string(str(value))

    return value

def clean_and_format_metrics(metrics):
    """
    Format the metrics dictionary by cleaning nested structures.
    """
    cleaned_metrics = {key: clean_value(value) for key, value in metrics.items()}
    return cleaned_metrics

def save_to_csv_incrementally(result, path):
    """
    Saves a single row of calibration results to a CSV file incrementally.
    
    Args:
        - result (dict): A single row of calibration results.
        - path (str): File path to save CSV.
    """
    # Clean and format metrics before saving
    cleaned_result = clean_and_format_metrics(result)

    # Convert the cleaned result to a DataFrame
    df = pd.DataFrame([cleaned_result])

    # Check if the file exists and is not empty to determine whether to include headers
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    logger.info(f"File exists: {os.path.exists(path)}, File size: {os.path.getsize(path) if os.path.exists(path) else 'N/A'}, Writing header: {write_header}")

    # Save with minimal quoting and no escape character
    df.to_csv(path, mode='a', header=write_header, index=False, quoting=csv.QUOTE_MINIMAL)
    logger.info(f"Saved result for {result.get('Dataset Name', 'Unknown')} - {result.get('Classification Model', 'Unknown')} (n={result.get('n', 'Unknown')})")

def fetch_completed_datasets(db: Session, table):
    """Fetch dataset names excluding specific ones, ensuring only those with 'completed' status remain."""
    try:
        datasets = (
            db.query(table.dataset_name)
            .group_by(table.dataset_name)
            .having(
                func.count(func.distinct(table.status)) == 1,
                func.max(table.status) == 'completed'
            )
            .all()
        )
        return [row.dataset_name for row in datasets]
    except Exception as e:
        logging.error(f"Error fetching completed datasets: {e}")
        return []

def extract_uncalibrated_data(db: Session, table, dataset_name: str):
    """Fetch uncalibrated probabilities and ground truth for each classification model before calibration."""
    try:
        models_data = (
            db.query(
                table.dataset_name,
                table.classification_model,
                table.uncalibrated_probs_cal_set,
                table.ground_truth_cal_set,
                table.uncalibrated_probs_test_set,
                table.ground_truth_test_set, 
                table.uncalibrated_train_loss, 
                table.uncalibrated_train_accuracy,
                table.uncalibrated_train_ece,
                table.uncalibrated_train_mce,
                table.uncalibrated_train_conf_ece,
                table.uncalibrated_train_f1_score_macro,
                table.uncalibrated_train_f1_score_micro,
                table.uncalibrated_train_f1_score_weighted,
                table.uncalibrated_train_recall_macro,
                table.uncalibrated_train_recall_micro,
                table.uncalibrated_train_recall_weighted,
                table.uncalibrated_train_precision_macro,
                table.uncalibrated_train_precision_micro,
                table.uncalibrated_train_precision_weighted,
                table.uncalibrated_train_brier_score
            )
            .filter(table.dataset_name == dataset_name, table.status == "completed")
            .distinct(table.dataset_name, table.classification_model)
            .all()
        )

        extracted_data = []
        for experiment in models_data:
            classification_model = experiment.classification_model

            # Ensure required data is available
            if not experiment.uncalibrated_probs_cal_set or not experiment.ground_truth_cal_set:
                logging.warning(f"Skipping {dataset_name} - {classification_model}: Missing calibration data.")
                continue
            if not experiment.uncalibrated_probs_test_set or not experiment.ground_truth_test_set:
                logging.warning(f"Skipping {dataset_name} - {classification_model}: Missing test data.")
                continue

            # Convert data to numpy arrays
            extracted_data.append({
                "dataset_name": dataset_name,
                "classification_model": classification_model,
                
                "uncalibrated_probs_cal_set": np.array(experiment.uncalibrated_probs_cal_set),
                "ground_truth_cal_set": np.array(experiment.ground_truth_cal_set),
                "uncalibrated_probs_test_set": np.array(experiment.uncalibrated_probs_test_set),
                "ground_truth_test_set": np.array(experiment.ground_truth_test_set),
                
                # Uncalibrated Metrics (train set)
                "uncalibrated_train_loss": experiment.uncalibrated_train_loss,
                "uncalibrated_train_accuracy": experiment.uncalibrated_train_accuracy,
                "uncalibrated_train_ece": experiment.uncalibrated_train_ece,
                "uncalibrated_train_mce": experiment.uncalibrated_train_mce,
                "uncalibrated_train_conf_ece": experiment.uncalibrated_train_conf_ece,
                "uncalibrated_train_f1_score_macro": experiment.uncalibrated_train_f1_score_macro,
                "uncalibrated_train_f1_score_micro": experiment.uncalibrated_train_f1_score_micro,
                "uncalibrated_train_f1_score_weighted": experiment.uncalibrated_train_f1_score_weighted,
                "uncalibrated_train_recall_macro": experiment.uncalibrated_train_recall_macro,
                "uncalibrated_train_recall_micro": experiment.uncalibrated_train_recall_micro,
                "uncalibrated_train_recall_weighted": experiment.uncalibrated_train_recall_weighted,
                "uncalibrated_train_precision_macro": experiment.uncalibrated_train_precision_macro,
                "uncalibrated_train_precision_micro": experiment.uncalibrated_train_precision_micro,
                "uncalibrated_train_precision_weighted": experiment.uncalibrated_train_precision_weighted,
                "uncalibrated_train_brier_score": experiment.uncalibrated_train_brier_score
            })
        return extracted_data

    except Exception as e:
        logging.error(f"Error extracting uncalibrated data for dataset {dataset_name}: {e}")
        db.rollback()  # Recover from error
        return []
    
def save_best_result(dataset_name, classification_model, 
                     n, best_config, 
                     uncalibrated_train_loss, uncalibrated_train_accuracy, uncalibrated_train_ece, uncalibrated_train_mce, uncalibrated_train_conf_ece, 
                     uncalibrated_train_f1_score_macro, uncalibrated_train_f1_score_micro, uncalibrated_train_f1_score_weighted, 
                     uncalibrated_train_recall_macro, uncalibrated_train_recall_micro, uncalibrated_train_recall_weighted, 
                     uncalibrated_train_precision_macro, uncalibrated_train_precision_micro, uncalibrated_train_precision_weighted, 
                     uncalibrated_train_brier_score, 
                     best_classification_metrics_calibrated, best_cal_metrics_calibrated,
                     best_classification_metrics_test, best_cal_metrics_test, 
                     output_path):
    """
    Save the best calibration result to a CSV file.
    """
    if best_config:
        try:
            best_result = {
                "Dataset Name": dataset_name,
                "Classification Model": classification_model,
                "n": n,
                "Best Calibrator": best_config.get("Calibration_Algorithm"),
                "Hyperparameters": str({k: v for k, v in best_config.items() if k != "Calibration_Algorithm"}),
                "uncalibrated_train_loss": uncalibrated_train_loss, 
                "uncalibrated_train_accuracy": uncalibrated_train_accuracy, 
                "uncalibrated_train_ece": uncalibrated_train_ece, 
                "uncalibrated_train_mce": uncalibrated_train_mce, 
                "uncalibrated_train_conf_ece": uncalibrated_train_conf_ece, 
                "uncalibrated_train_f1_score_macro": uncalibrated_train_f1_score_macro, 
                "uncalibrated_train_f1_score_micro": uncalibrated_train_f1_score_micro, 
                "uncalibrated_train_f1_score_weighted": uncalibrated_train_f1_score_weighted, 
                "uncalibrated_train_recall_macro": uncalibrated_train_recall_macro, 
                "uncalibrated_train_recall_micro": uncalibrated_train_recall_micro, 
                "uncalibrated_train_recall_weighted": uncalibrated_train_recall_weighted, 
                "uncalibrated_train_precision_macro": uncalibrated_train_precision_macro, 
                "uncalibrated_train_precision_micro": uncalibrated_train_precision_micro, 
                "uncalibrated_train_precision_weighted": uncalibrated_train_precision_weighted, 
                "uncalibrated_train_brier_score": uncalibrated_train_brier_score,

                "calibrated_loss": best_classification_metrics_calibrated["loss"], 
                "calibrated_accuracy": best_classification_metrics_calibrated["accuracy"], 
                "calibrated_f1_score_macro": best_classification_metrics_calibrated["f1_macro"], 
                "calibrated_f1_score_micro": best_classification_metrics_calibrated["f1_micro"], 
                "calibrated_f1_score_weighted": best_classification_metrics_calibrated["f1_weighted"], 
                "calibrated_recall_macro": best_classification_metrics_calibrated["recall_macro"], 
                "calibrated_recall_micro": best_classification_metrics_calibrated["recall_micro"], 
                "calibrated_recall_weighted": best_classification_metrics_calibrated["recall_weighted"], 
                "calibrated_precision_macro": best_classification_metrics_calibrated["precision_macro"], 
                "calibrated_precision_micro": best_classification_metrics_calibrated["precision_micro"], 
                "calibrated_precision_weighted": best_classification_metrics_calibrated["precision_weighted"], 
                "calibrated_ece": best_cal_metrics_calibrated["ece"], 
                "calibrated_mce": best_cal_metrics_calibrated["mce"], 
                "calibrated_conf_ece": best_cal_metrics_calibrated["conf_ece"], 
                "calibrated_brier_score": best_cal_metrics_calibrated["brier_score"],

                "test_loss": best_classification_metrics_test["loss"],
                "test_accuracy": best_classification_metrics_test["accuracy"],
                "test_f1_score_macro": best_classification_metrics_test["f1_macro"],
                "test_f1_score_micro": best_classification_metrics_test["f1_micro"],
                "test_f1_score_weighted": best_classification_metrics_test["f1_weighted"],
                "test_recall_macro": best_classification_metrics_test["recall_macro"],
                "test_recall_micro": best_classification_metrics_test["recall_micro"],
                "test_recall_weighted": best_classification_metrics_test["recall_weighted"],
                "test_precision_macro": best_classification_metrics_test["precision_macro"],
                "test_precision_micro": best_classification_metrics_test["precision_micro"],
                "test_precision_weighted": best_classification_metrics_test["precision_weighted"],
                "test_ece": best_cal_metrics_test["ece"],
                "test_mce": best_cal_metrics_test["mce"],
                "test_conf_ece": best_cal_metrics_test["conf_ece"],
                "test_brier_score": best_cal_metrics_test["brier_score"],
            }
            save_to_csv_incrementally(best_result, output_path)
        except Exception as e:
            logger.error(f"Error saving best result for {dataset_name} - {classification_model}: {e}")

def get_hashed_seed(dataset, model):
    """Generate a consistent random seed from dataset and model."""
    return int(hashlib.sha256(f"{dataset}_{model}".encode()).hexdigest(), 16) % (10 ** 8)

def process_random_baseline(db: Session, table, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    completed_datasets = fetch_completed_datasets(db, table)

    all_combinations = get_all_calibrator_combinations()
    all_combinations = sorted(all_combinations, key=lambda x: json.dumps(x, sort_keys=True))

    for dataset_name in completed_datasets:
        dataset_rows = extract_uncalibrated_data(db, table, dataset_name)

        if not dataset_rows:
            continue
        
        for row in dataset_rows:
            classification_model = row["classification_model"]

            seed = get_hashed_seed(dataset_name,classification_model)

            np.random.seed(seed)
            random.seed(seed)  # in case other code uses global RNG

            uncalibrated_probs_valid = row["uncalibrated_probs_cal_set"]
            uncalibrated_probs_test = row["uncalibrated_probs_test_set"]
            y_cal = row["ground_truth_cal_set"]
            y_test = row["ground_truth_test_set"]

            # Training metrics
            uncalibrated_train_loss = row["uncalibrated_train_loss"]
            uncalibrated_train_accuracy = row["uncalibrated_train_accuracy"]
            uncalibrated_train_ece = row["uncalibrated_train_ece"]
            uncalibrated_train_mce = row["uncalibrated_train_mce"]
            uncalibrated_train_conf_ece = row["uncalibrated_train_conf_ece"]
            uncalibrated_train_f1_score_macro = row["uncalibrated_train_f1_score_macro"]
            uncalibrated_train_f1_score_micro = row["uncalibrated_train_f1_score_micro"]
            uncalibrated_train_f1_score_weighted = row["uncalibrated_train_f1_score_weighted"]
            uncalibrated_train_recall_macro = row["uncalibrated_train_recall_macro"]
            uncalibrated_train_recall_micro = row["uncalibrated_train_recall_micro"]
            uncalibrated_train_recall_weighted = row["uncalibrated_train_recall_weighted"]
            uncalibrated_train_precision_macro = row["uncalibrated_train_precision_macro"]
            uncalibrated_train_precision_micro = row["uncalibrated_train_precision_micro"],
            uncalibrated_train_precision_weighted = row["uncalibrated_train_precision_weighted"]
            uncalibrated_train_brier_score = row["uncalibrated_train_brier_score"]

            best_config = {10: None, 30: None, 50: None}
            best_ece = {10: float("inf"), 30: float("inf"), 50: float("inf")}
            best_classification_metrics_calibrated = {10: None, 30: None, 50: None}
            best_cal_metrics_calibrated = {10: None, 30: None, 50: None}
            best_classification_metrics_test = {10: None, 30: None, 50: None}
            best_cal_metrics_calibrated_test = {10: None, 30: None, 50: None}
            random_combinations = get_n_random_calibration_combinations(all_combinations, 50, seed)

            for i, calibrator_dict in enumerate(random_combinations):
                try:
                    # Apply calibration and get calibrated probabilities
                    calibrated_probs_test, calibrated_probs_valid = apply_calibration_with_cv(
                        calibrator_dict, uncalibrated_probs_valid, y_cal, uncalibrated_probs_test
                    )

                    # Calculate metrics for calibrated validation set
                    cal_metrics_calibrated = compute_calibration_metrics(
                        calibrated_probs_valid, np.argmax(calibrated_probs_valid, axis=1), y_cal, 
                        ['ece', 'mce', 'conf_ece', 'brier_score']
                    )
                    classification_metrics_calibrated = compute_classification_metrics(
                        y_cal, np.argmax(calibrated_probs_valid, axis=1), calibrated_probs_valid
                    )

                    # Print the current trial number and ECE
                    print(f"Trial {i+1}/50 for dataset {dataset_name} - {classification_model} - {calibrator_dict} - seed={seed}: n={i+1}, ECE={cal_metrics_calibrated['ece']}")

                    # Calculate metrics for calibrated test set
                    classification_metrics_test = compute_classification_metrics(
                        y_test, np.argmax(calibrated_probs_test, axis=1), calibrated_probs_test
                    )
                    cal_metrics_test = compute_calibration_metrics(
                        calibrated_probs_test, np.argmax(calibrated_probs_test, axis=1), y_test, 
                        ['ece', 'mce', 'conf_ece', 'brier_score']
                    )

                    # Track the best configuration after 10, 30, and 50 combinations
                    for n in [10, 30, 50]:
                        if i < n and cal_metrics_calibrated["ece"] < best_ece[n]:
                            best_ece[n] = cal_metrics_calibrated["ece"]
                            best_config[n] = calibrator_dict
                            best_classification_metrics_calibrated[n] = classification_metrics_calibrated
                            best_cal_metrics_calibrated[n] = cal_metrics_calibrated
                            best_classification_metrics_test[n] = classification_metrics_test
                            best_cal_metrics_calibrated_test[n] = cal_metrics_test
                            logger.info(f"New best configuration after {i+1}/{n} combinations for {dataset_name} - {classification_model}: ECE = {cal_metrics_calibrated['ece']}")
                            
                except Exception as e:
                    logger.warning(f"Skipping calibrator due to error for dataset '{dataset_name}', model '{classification_model}', calibrator {calibrator_dict}: {e}")
                
                # Always check after the loop iteration, even if it failed
                if i + 1 in [10, 30, 50]:
                    if best_config[i + 1] is not None:
                        save_best_result(
                            dataset_name, classification_model, i + 1, best_config[i + 1],
                            uncalibrated_train_loss, uncalibrated_train_accuracy, uncalibrated_train_ece, uncalibrated_train_mce, uncalibrated_train_conf_ece, 
                            uncalibrated_train_f1_score_macro, uncalibrated_train_f1_score_micro, uncalibrated_train_f1_score_weighted, 
                            uncalibrated_train_recall_macro, uncalibrated_train_recall_micro, uncalibrated_train_recall_weighted, 
                            uncalibrated_train_precision_macro, uncalibrated_train_precision_micro, uncalibrated_train_precision_weighted, 
                            uncalibrated_train_brier_score, 
                            best_classification_metrics_calibrated[i + 1], best_cal_metrics_calibrated[i + 1], 
                            best_classification_metrics_test[i + 1], best_cal_metrics_calibrated_test[i + 1],
                            output_path
                        )
                    else:
                        logger.warning(f"No valid calibrator found after {i + 1} trials for dataset '{dataset_name}', model '{classification_model}'")
