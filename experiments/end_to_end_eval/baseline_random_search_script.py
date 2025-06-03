import logging
import warnings
import numpy as np
import pandas as pd
import os
import random
import json
import ast
import csv
import hashlib
from sqlalchemy.orm import Session
from sqlalchemy import func
from sklearn.model_selection import StratifiedKFold
import traceback

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from smartcal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from smartcal.utils.calibrators_hyperparameters import get_all_calibrator_combinations
from smartcal.utils.cal_metrics import compute_calibration_metrics
from smartcal.utils.classification_metrics import compute_classification_metrics
from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment_V2


# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#logging.basicConfig(level=logging.WARNING)  # This line configures the logging module to display only warnings or more critical messages in the terminal.
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="The y_pred values do not sum to one. Make sure to pass probabilities.")

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

def load_class_distributions(csv_path=config_manager.all_class_distributions_file):
    """
    Load class distributions from CSV file and convert to probability distributions.
    Returns a dictionary mapping metrics to algorithm probabilities.
    """
    df = pd.read_csv(csv_path)
    distributions = {}
    
    for metric in df['Metric'].unique():
        metric_data = df[df['Metric'] == metric]
        total = metric_data['Count'].sum()
        probabilities = metric_data.set_index('Class')['Count'] / total
        distributions[metric] = probabilities.to_dict()
    
    return distributions

def get_n_random_calibration_combinations(combinations_dicts, n=50, seed=31, metric_type="ECE"):
    """
    Get n random calibration combinations based on class distributions.
    
    Args:
        combinations_dicts: List of all possible calibration combinations
        n: Number of combinations to return
        seed: Random seed for reproducibility
        metric_type: The metric type to use for distribution sampling (ECE, MCE, ConfECE, brier_score, log_loss)
    """
    rng = random.Random(seed)  # local RNG instance
    n = min(n, len(combinations_dicts))
    
    # Load class distributions
    distributions = load_class_distributions()
    
    # Get probability distribution for the specified metric
    if metric_type not in distributions:
        logger.warning(f"Metric {metric_type} not found in distributions, using uniform sampling")
        return get_n_random_calibration_combinations_uniform(combinations_dicts, n, seed)
    
    metric_distribution = distributions[metric_type]
    logger.info(f"\nMetric: {metric_type}")
    logger.info("Algorithm probabilities:")
    for algo, prob in sorted(metric_distribution.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {algo}: {prob:.3f}")
    
    # Group combinations by algorithm type
    combinations_by_algo = {}
    for combo in combinations_dicts:
        algo = combo.get("Calibration_Algorithm")
        if algo not in combinations_by_algo:
            combinations_by_algo[algo] = []
        combinations_by_algo[algo].append(combo)
    
    # Log available combinations per algorithm
    logger.info("\nAvailable combinations per algorithm:")
    for algo in sorted(combinations_by_algo.keys()):
        logger.info(f"  {algo}: {len(combinations_by_algo[algo])} combinations available")
    
    # Initialize tracking
    selected_combinations = []
    algo_counts = {algo: 0 for algo in metric_distribution.keys()}
    
    # Adjust probabilities for sampling
    total_prob = sum(metric_distribution.values())
    adjusted_probs = {algo: prob/total_prob for algo, prob in metric_distribution.items()}
    
    # Sample combinations based on the distribution
    while len(selected_combinations) < n:
        # Sample algorithm based on distribution
        algo = rng.choices(
            list(adjusted_probs.keys()),
            weights=list(adjusted_probs.values()),
            k=1
        )[0]
        
        # If algorithm exists in our combinations, sample from it
        if algo in combinations_by_algo and combinations_by_algo[algo]:
            combo = rng.choice(combinations_by_algo[algo])
            selected_combinations.append(combo)
            algo_counts[algo] += 1
            combinations_by_algo[algo].remove(combo)
    
    # Log the final distribution of selected algorithms
    logger.info("\nFinal selected combinations distribution:")
    total_selected = sum(algo_counts.values())
    for algo, count in sorted(algo_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {algo}: {count} ({count/total_selected:.3f})")
    
    return selected_combinations

def get_n_random_calibration_combinations_uniform(combinations_dicts, n=50, seed=31):
    """Original uniform random sampling function as fallback."""
    rng = random.Random(seed)
    n = min(n, len(combinations_dicts))
    all_combinations = [(rng.random(), combination) for combination in combinations_dicts]
    all_combinations.sort(key=lambda x: x[0])
    return [combination for _, combination in all_combinations[:n]]

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
        # If parsing was successful and it's a list or tuple, convert to a proper list
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
                func.max(table.status) == Experiment_Status_Enum.COMPLETED.value
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
            .filter(table.dataset_name == dataset_name, table.status == Experiment_Status_Enum.COMPLETED.value)
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
                     output_path,
                     metric_type):
    """
    Save the best calibration result to a CSV file.
    """
    if best_config:
        try:
            best_result = {
                "Dataset Name": dataset_name,
                "Classification Model": classification_model,
                "n": n,
                "Calibration Metric": metric_type,
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
            logger.error(f"Error saving best result for {dataset_name} - {classification_model} - {metric_type}: {e}")

def get_hashed_seed(dataset, model):
    """Generate a consistent random seed from dataset and model."""
    return int(hashlib.sha256(f"{dataset}_{model}".encode()).hexdigest(), 16) % (10 ** 8)

def process_random_baseline(db: Session, table, output_path, checkpoint_steps=config_manager.num_itr_eval):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Get all possible combinations first
    all_combinations = get_all_calibrator_combinations()
    all_combinations = sorted(all_combinations, key=lambda x: json.dumps(x, sort_keys=True))
    
    logger.info("\n" + "="*80)
    logger.info("INITIAL SETUP")
    logger.info("="*80)
    
    # Count combinations per algorithm
    algo_combinations = {}
    for combo in all_combinations:
        algo = combo.get("Calibration_Algorithm")
        if algo not in algo_combinations:
            algo_combinations[algo] = []
        algo_combinations[algo].append(combo)
    
    logger.info("\nNumber of combinations per calibration algorithm:")
    for algo in sorted(algo_combinations.keys()):
        logger.info(f"  {algo}: {len(algo_combinations[algo])} combinations")
    
    # Log available algorithms and their hyperparameters
    logger.info("\nAvailable Calibration Algorithms and their hyperparameters:")
    algos = {}
    for combo in all_combinations:
        algo = combo.get("Calibration_Algorithm")
        if algo not in algos:
            algos[algo] = set()
        for key, value in combo.items():
            if key != "Calibration_Algorithm":
                algos[algo].add(f"{key}: {type(value).__name__}")
    
    for algo in sorted(algos.keys()):
        logger.info(f"\n{algo}:")
        for param in sorted(algos[algo]):
            logger.info(f"  {param}")
    
    logger.info(f"\nTotal number of possible combinations: {len(all_combinations)}")
    
    # Get completed datasets
    completed_datasets = fetch_completed_datasets(db, table)
    logger.info(f"\nFound {len(completed_datasets)} completed datasets:")
    for dataset in sorted(completed_datasets):
        logger.info(f"  {dataset}")

    # Define metrics to track
    metrics = ["ECE", "MCE", "ConfECE", "brier_score", "log_loss"]
    metric_mapping = {
        "ECE": "ece",
        "MCE": "mce",
        "ConfECE": "conf_ece",
        "brier_score": "brier_score",
        "log_loss": "loss"
    }
    
    logger.info("\nMetrics to track:")
    for metric in metrics:
        logger.info(f"  {metric}")

    # Load class distributions once at the start
    distributions = load_class_distributions()
    logger.info("\nClass distributions from CSV:")
    for metric in metrics:
        if metric in distributions:
            logger.info(f"\n{metric} distribution:")
            for algo, prob in sorted(distributions[metric].items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {algo}: {prob:.3f}")

    logger.info("\n" + "="*80)
    logger.info("STARTING EXPERIMENTS")
    logger.info("="*80)

    for dataset_name in completed_datasets:
        logger.info(f"\nProcessing dataset: {dataset_name}")
        dataset_rows = extract_uncalibrated_data(db, table, dataset_name)

        if not dataset_rows:
            logger.warning(f"No data found for dataset: {dataset_name}")
            continue
        
        for row in dataset_rows:
            classification_model = row["classification_model"]
            logger.info(f"\nProcessing model: {classification_model}")

            seed = get_hashed_seed(dataset_name, classification_model)
            logger.info(f"Using seed: {seed}")

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
            uncalibrated_train_precision_micro = row["uncalibrated_train_precision_micro"]
            uncalibrated_train_precision_weighted = row["uncalibrated_train_precision_weighted"]
            uncalibrated_train_brier_score = row["uncalibrated_train_brier_score"]

            # Initialize tracking for each metric and checkpoint
            best_configs = {metric: {n: None for n in checkpoint_steps} for metric in metrics}
            best_scores = {metric: {n: float("inf") for n in checkpoint_steps} for metric in metrics}
            best_classification_metrics_calibrated = {metric: {n: None for n in checkpoint_steps} for metric in metrics}
            best_cal_metrics_calibrated = {metric: {n: None for n in checkpoint_steps} for metric in metrics}
            best_classification_metrics_test = {metric: {n: None for n in checkpoint_steps} for metric in metrics}
            best_cal_metrics_calibrated_test = {metric: {n: None for n in checkpoint_steps} for metric in metrics}

            # Get random combinations for each metric using their respective distributions
            random_combinations_by_metric = {}
            logger.info(f"\n{'='*80}")
            logger.info(f"Generating combinations for {dataset_name} - {classification_model}")
            logger.info(f"{'='*80}")
            
            for metric in metrics:
                logger.info(f"\n{'-'*40}")
                logger.info(f"Metric: {metric}")
                logger.info(f"{'-'*40}")
                
                # Get and log the distribution for this metric
                if metric in distributions:
                    logger.info("\nTarget distribution:")
                    for algo, prob in sorted(distributions[metric].items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"  {algo}: {prob:.3f}")
                
                # Get combinations for this metric
                random_combinations_by_metric[metric] = get_n_random_calibration_combinations(
                    all_combinations, max(checkpoint_steps), seed, metric
                )
                
                # Log the actual combinations selected
                logger.info("\nActual combinations selected:")
                algo_counts = {}
                for combo in random_combinations_by_metric[metric]:
                    algo = combo.get("Calibration_Algorithm")
                    algo_counts[algo] = algo_counts.get(algo, 0) + 1
                
                for algo, count in sorted(algo_counts.items(), key=lambda x: x[1], reverse=True):
                    target_prob = distributions[metric].get(algo, 0)
                    logger.info(f"  {algo}: {count} combinations (target: {target_prob:.3f})")
                
                logger.info(f"Total combinations for {metric}: {len(random_combinations_by_metric[metric])}")

            # Process each metric's combinations
            for metric in metrics:
                random_combinations = random_combinations_by_metric[metric]
                logger.info(f"\n{'-'*40}")
                logger.info(f"Processing combinations for metric: {metric}")
                logger.info(f"{'-'*40}")
                
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

                        # Calculate metrics for calibrated test set
                        classification_metrics_test = compute_classification_metrics(
                            y_test, np.argmax(calibrated_probs_test, axis=1), calibrated_probs_test
                        )
                        cal_metrics_test = compute_calibration_metrics(
                            calibrated_probs_test, np.argmax(calibrated_probs_test, axis=1), y_test, 
                            ['ece', 'mce', 'conf_ece', 'brier_score']
                        )

                        # Track the best configuration for each checkpoint
                        for n in checkpoint_steps:
                            if i < n:
                                metric_key = metric_mapping[metric]
                                
                                # Use classification metrics for loss
                                if metric == "log_loss":
                                    current_score = classification_metrics_calibrated[metric_key]
                                else:
                                    current_score = cal_metrics_calibrated[metric_key]
                                
                                # Handle ConfECE tuple by taking first value
                                if metric == "ConfECE" and isinstance(current_score, tuple):
                                    current_score = current_score[0]  # Take first value from tuple
                                
                                if current_score < best_scores[metric][n]:
                                    best_scores[metric][n] = current_score
                                    best_configs[metric][n] = calibrator_dict
                                    best_classification_metrics_calibrated[metric][n] = classification_metrics_calibrated
                                    best_cal_metrics_calibrated[metric][n] = cal_metrics_calibrated
                                    best_classification_metrics_test[metric][n] = classification_metrics_test
                                    best_cal_metrics_calibrated_test[metric][n] = cal_metrics_test
                                    logger.info(f"New best configuration for {metric} after {i+1}/{n} combinations for {dataset_name} - {classification_model}: {metric} = {current_score}")
                        
                    except Exception as e:
                        error_details = traceback.format_exc()
                        logger.warning(f"Skipping calibrator due to error for dataset '{dataset_name}', model '{classification_model}', calibrator {calibrator_dict}:\nError details:\n{error_details}")
                    
                    # Save results for each checkpoint
                    if i + 1 in checkpoint_steps:
                        if best_configs[metric][i + 1] is not None:
                            save_best_result(
                                dataset_name, classification_model, i + 1, best_configs[metric][i + 1],
                                uncalibrated_train_loss, uncalibrated_train_accuracy, uncalibrated_train_ece, uncalibrated_train_mce, uncalibrated_train_conf_ece, 
                                uncalibrated_train_f1_score_macro, uncalibrated_train_f1_score_micro, uncalibrated_train_f1_score_weighted, 
                                uncalibrated_train_recall_macro, uncalibrated_train_recall_micro, uncalibrated_train_recall_weighted, 
                                uncalibrated_train_precision_macro, uncalibrated_train_precision_micro, uncalibrated_train_precision_weighted, 
                                uncalibrated_train_brier_score, 
                                best_classification_metrics_calibrated[metric][i + 1], best_cal_metrics_calibrated[metric][i + 1], 
                                best_classification_metrics_test[metric][i + 1], best_cal_metrics_calibrated_test[metric][i + 1],
                                output_path,
                                metric
                            )
                        else:
                            logger.warning(f"No valid calibrator found for {metric} after {i + 1} trials for dataset '{dataset_name}', model '{classification_model}'")

if __name__ == "__main__":
    with SessionLocal() as db:
        process_random_baseline(db, BenchmarkingExperiment_V2, "experiments/end_to_end_eval/Results/baseline_random_search_results.csv")
