import csv
import os
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
import logging
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment_V2
from smartcal.bayesian_optimization.calibrators_bayesian_optimization import CalibrationOptimizer
from smartcal.meta_model.meta_model import MetaModel
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()


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


def fetch_cal_and_test_data(db, table, dataset_name):
    """Fetch calibration and test set data (ground truth labels and predicted probabilities)
    for a specific dataset, grouped by dataset name and classification model."""
    try:
        logging.info(f"Fetching data for dataset: {dataset_name}")
        # Query all relevant data including performance metrics for each calibrator
        calibration_data = (
            db.query(
                table.dataset_name,
                table.classification_model,

                table.uncalibrated_train_ece,
                table.uncalibrated_train_mce,
                table.uncalibrated_train_conf_ece,
                table.uncalibrated_train_f1_score_micro,
                table.uncalibrated_train_f1_score_macro,
                table.uncalibrated_train_recall_micro,
                table.uncalibrated_train_recall_macro,
                table.uncalibrated_train_precision_micro,
                table.uncalibrated_train_precision_macro,
                table.uncalibrated_train_brier_score,
                table.uncalibrated_train_calibration_curve_mean_predicted_probs,
                table.uncalibrated_train_calibration_curve_true_probs,
                table.uncalibrated_train_calibration_num_bins,

                # Uncalibrated Metrics (validation)
                table.uncalibrated_cal_recall_micro,
                table.uncalibrated_cal_recall_macro,
                table.uncalibrated_cal_precision_micro,
                table.uncalibrated_cal_precision_macro,
                table.uncalibrated_cal_f1_score_micro,
                table.uncalibrated_cal_f1_score_macro,
                table.uncalibrated_cal_ece,
                table.uncalibrated_cal_mce,
                table.uncalibrated_cal_conf_ece,
                table.uncalibrated_cal_brier_score,
                table.uncalibrated_cal_calibration_curve_mean_predicted_probs,
                table.uncalibrated_cal_calibration_curve_true_probs,
                table.uncalibrated_cal_calibration_num_bins,

                # Uncalibrated Metrics (test)
                table.uncalibrated_test_ece,
                table.uncalibrated_test_mce,
                table.uncalibrated_test_conf_ece,
                table.uncalibrated_test_brier_score,
                table.uncalibrated_test_calibration_curve_mean_predicted_probs,
                table.uncalibrated_test_calibration_curve_true_probs,
                table.uncalibrated_test_calibration_num_bins,

                table.ground_truth_cal_set,
                table.uncalibrated_probs_cal_set,
                table.ground_truth_test_set,
                table.uncalibrated_probs_test_set
            )
            .filter(
                table.dataset_name == dataset_name,
                table.status == Experiment_Status_Enum.COMPLETED.value
            )
            .all()
        )

        logging.info(f"Found {len(calibration_data)} records for dataset {dataset_name}")

        # Group data by dataset and model
        results = {}
        for data in calibration_data:
            key = (data.dataset_name, data.classification_model)

            if key not in results:
                try:
                    # Convert string representations to numpy arrays if needed
                    ground_truth_cal = np.array(eval(data.ground_truth_cal_set)) if isinstance(
                        data.ground_truth_cal_set, str) else np.array(data.ground_truth_cal_set)
                    uncal_probs_cal = np.array(eval(data.uncalibrated_probs_cal_set)) if isinstance(
                        data.uncalibrated_probs_cal_set, str) else np.array(data.uncalibrated_probs_cal_set)
                    ground_truth_test = np.array(eval(data.ground_truth_test_set)) if isinstance(
                        data.ground_truth_test_set, str) else np.array(data.ground_truth_test_set)
                    uncal_probs_test = np.array(eval(data.uncalibrated_probs_test_set)) if isinstance(
                        data.uncalibrated_probs_test_set, str) else np.array(data.uncalibrated_probs_test_set)

                    results[key] = {
                        "dataset_name": data.dataset_name,
                        "classification_model": data.classification_model,
                        "calibration_set_true_labels": ground_truth_cal,
                        "calibration_set_predicted_probabilities": uncal_probs_cal,

                        "test_set_true_labels": ground_truth_test,
                        "test_set_predicted_probabilities": uncal_probs_test,
                        "calibrator_performances": {},

                        "train_ece": data.uncalibrated_train_ece,
                        "train_mce": data.uncalibrated_train_mce,
                        "train_conf_ece": data.uncalibrated_train_conf_ece,
                        "train_f1_score_micro": data.uncalibrated_train_f1_score_micro,
                        "train_f1_score_macro": data.uncalibrated_train_f1_score_macro,
                        "train_recall_micro": data.uncalibrated_train_recall_micro,
                        "train_recall_macro": data.uncalibrated_train_recall_macro,
                        "train_precision_micro": data.uncalibrated_train_precision_micro,
                        "train_precision_macro": data.uncalibrated_train_precision_macro,
                        "train_brier_score": data.uncalibrated_train_brier_score,
                        "train_calibration_curve_mean_predicted_probs": data.uncalibrated_train_calibration_curve_mean_predicted_probs,
                        "train_calibration_curve_true_probs": data.uncalibrated_train_calibration_curve_true_probs,
                        "train_calibration_num_bins": data.uncalibrated_train_calibration_num_bins,

                        # Uncalibrated Metrics (validation)
                        "uncalibrated_cal_recall_micro": data.uncalibrated_cal_recall_micro,
                        "uncalibrated_cal_recall_macro": data.uncalibrated_cal_recall_macro,
                        "uncalibrated_cal_precision_micro": data.uncalibrated_cal_precision_micro,
                        "uncalibrated_cal_precision_macro": data.uncalibrated_cal_precision_macro,
                        "uncalibrated_cal_f1_score_micro": data.uncalibrated_cal_f1_score_micro,
                        "uncalibrated_cal_f1_score_macro": data.uncalibrated_cal_f1_score_macro,
                        "uncalibrated_cal_ece": data.uncalibrated_cal_ece,
                        "uncalibrated_cal_mce": data.uncalibrated_cal_mce,
                        "uncalibrated_cal_conf_ece": data.uncalibrated_cal_conf_ece,
                        "uncalibrated_cal_brier_score": data.uncalibrated_cal_brier_score,
                        "uncalibrated_cal_calibration_curve_mean_predicted_probs": data.uncalibrated_cal_calibration_curve_mean_predicted_probs,
                        "uncalibrated_cal_calibration_curve_true_probs": data.uncalibrated_cal_calibration_curve_true_probs,
                        "uncalibrated_cal_calibration_num_bins": data.uncalibrated_cal_calibration_num_bins,

                        # Uncalibrated Metrics (test)
                        "uncalibrated_test_ece": data.uncalibrated_test_ece,
                        "uncalibrated_test_mce": data.uncalibrated_test_mce,
                        "uncalibrated_test_conf_ece": data.uncalibrated_test_conf_ece,
                        "uncalibrated_test_brier_score": data.uncalibrated_test_brier_score,
                        "uncalibrated_test_calibration_curve_mean_predicted_probs": data.uncalibrated_test_calibration_curve_mean_predicted_probs,
                        "uncalibrated_test_calibration_curve_true_probs": data.uncalibrated_test_calibration_curve_true_probs,
                        "uncalibrated_test_calibration_num_bins": data.uncalibrated_test_calibration_num_bins
                    }
                except Exception as e:
                    logging.error(f"Error processing data for {key}: {e}")
                    continue

        return list(results.values())

    except Exception as e:
        logging.error(f"Error fetching data for dataset {dataset_name}: {e}", exc_info=True)
        return []

def combine_smartcal_csvs():
    # Define the base directory (Results folder from root directory)
    results_dir = Path("experiments/end_to_end_eval/Results")
    
    # List of CSV files to combine
    csv_files = [
        "smartcal_results_brier_score.csv",
        "smartcal_results_ConfECE.csv", 
        "smartcal_results_ECE.csv",
        "smartcal_results_log_loss.csv",
        "smartcal_results_MCE.csv"
    ]
    
    # Check if all files exist
    missing_files = []
    for file in csv_files:
        file_path = results_dir / file
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print(f"Error: The following files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("Starting to combine CSV files...")
    
    # Read and combine all CSV files
    combined_df = pd.DataFrame()
    
    for i, file in enumerate(csv_files):
        file_path = results_dir / file
        print(f"Reading {file}...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"  - Loaded {len(df)} rows from {file}")
            
            if i == 0:
                combined_df = df
            else:
                # Concatenate dataframes, handling different columns gracefully
                combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
            return False
    
    # Save the combined dataframe
    output_path = results_dir / "smartcal_results.csv"
    print(f"\nSaving combined results to {output_path}...")
    
    try:
        combined_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(combined_df)} rows to {output_path}")
    except Exception as e:
        print(f"Error saving combined file: {e}")
        return False
    
    # Remove the original files
    print("\nRemoving original files...")
    removed_files = []
    failed_removals = []
    
    for file in csv_files:
        file_path = results_dir / file
        try:
            os.remove(file_path)
            removed_files.append(file)
            print(f"  - Removed {file}")
        except Exception as e:
            failed_removals.append((file, str(e)))
            print(f"  - Failed to remove {file}: {e}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Combined {len(csv_files)} CSV files into smartcal_results.csv")
    print(f"Total rows in combined file: {len(combined_df)}")
    print(f"Successfully removed {len(removed_files)} original files")
    
    if failed_removals:
        print(f"Failed to remove {len(failed_removals)} files:")
        for file, error in failed_removals:
            print(f"  - {file}: {error}")
        return False
    
    print("All operations completed successfully!")
    return True

# Output folder
output_folder = 'experiments/end_to_end_eval/Results'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Iterations to run
iterations_list = config_manager.num_itr_eval

# Loop through each metric
for current_metric in config_manager.supported_metrics:
    logging.info(f"\n{'=' * 80}\nRunning with metric: {current_metric}\n{'=' * 80}")

    # Create a new MetaModel with the current metric
    meta_model = MetaModel(
        top_n=config_manager.meta_model_k,
        metric=current_metric
    )

    # Create a new optimizer with the updated meta_model
    optimizer = CalibrationOptimizer(meta_model=meta_model)

    # Define output filename for this metric
    output_filename = os.path.join(output_folder, f'smartcal_results_{current_metric}.csv')

    # Flag to check if header needs to be written
    write_header = not os.path.exists(output_filename)

    # Open the CSV file in append mode for this metric
    with open(output_filename, mode='a', newline='') as f, SessionLocal() as db:
        # Prepare CSV writer
        fieldnames = None
        writer = None

        # Fetch completed datasets
        datasets = fetch_completed_datasets(db, BenchmarkingExperiment_V2)

        # Iterate through different iteration counts
        for itr in iterations_list:
            logging.info(f"Running optimization with {itr} iterations for metric {current_metric}...")

            for dataset_name in datasets:
                results = fetch_cal_and_test_data(db, BenchmarkingExperiment_V2, dataset_name)

                for dataset_result in results:
                    # Extract model name for clarity
                    model_name = dataset_result['classification_model']

                    # Run optimization for the current model, using the current metric
                    optimization_results = optimizer.run_optimization(
                        dataset_result, total_iterations=itr, metric=current_metric
                    )

                    # Add dataset name and model name to the results
                    optimization_results['dataset_name'] = dataset_name
                    optimization_results['model_name'] = model_name
                    optimization_results['Total_Iterations'] = itr
                    optimization_results['metric_used'] = current_metric

                    # Add the fields from dataset_result
                    optimization_results['train_ece'] = dataset_result['train_ece']
                    optimization_results['train_mce'] = dataset_result['train_mce']
                    optimization_results['train_conf_ece'] = dataset_result['train_conf_ece']
                    optimization_results['train_f1_score_micro'] = dataset_result['train_f1_score_micro']
                    optimization_results['train_f1_score_macro'] = dataset_result['train_f1_score_macro']
                    optimization_results['train_recall_micro'] = dataset_result['train_recall_micro']
                    optimization_results['train_recall_macro'] = dataset_result['train_recall_macro']
                    optimization_results['train_precision_micro'] = dataset_result['train_precision_micro']
                    optimization_results['train_precision_macro'] = dataset_result['train_precision_macro']
                    optimization_results['train_brier_score'] = dataset_result['train_brier_score']
                    optimization_results['train_calibration_curve_mean_predicted_probs'] = dataset_result[
                        'train_calibration_curve_mean_predicted_probs']
                    optimization_results['train_calibration_curve_true_probs'] = dataset_result[
                        'train_calibration_curve_true_probs']
                    optimization_results['train_calibration_num_bins'] = dataset_result['train_calibration_num_bins']

                    # Add the validation and test uncrossed metrics
                    optimization_results['uncalibrated_cal_recall_micro'] = dataset_result[
                        'uncalibrated_cal_recall_micro']
                    optimization_results['uncalibrated_cal_recall_macro'] = dataset_result[
                        'uncalibrated_cal_recall_macro']
                    optimization_results['uncalibrated_cal_precision_micro'] = dataset_result[
                        'uncalibrated_cal_precision_micro']
                    optimization_results['uncalibrated_cal_precision_macro'] = dataset_result[
                        'uncalibrated_cal_precision_macro']
                    optimization_results['uncalibrated_cal_f1_score_micro'] = dataset_result[
                        'uncalibrated_cal_f1_score_micro']
                    optimization_results['uncalibrated_cal_f1_score_macro'] = dataset_result[
                        'uncalibrated_cal_f1_score_macro']
                    optimization_results['uncalibrated_cal_ece'] = dataset_result['uncalibrated_cal_ece']
                    optimization_results['uncalibrated_cal_mce'] = dataset_result['uncalibrated_cal_mce']
                    optimization_results['uncalibrated_cal_conf_ece'] = dataset_result['uncalibrated_cal_conf_ece']
                    optimization_results['uncalibrated_cal_brier_score'] = dataset_result[
                        'uncalibrated_cal_brier_score']
                    optimization_results['uncalibrated_cal_calibration_curve_mean_predicted_probs'] = dataset_result[
                        'uncalibrated_cal_calibration_curve_mean_predicted_probs']
                    optimization_results['uncalibrated_cal_calibration_curve_true_probs'] = dataset_result[
                        'uncalibrated_cal_calibration_curve_true_probs']
                    optimization_results['uncalibrated_cal_calibration_num_bins'] = dataset_result[
                        'uncalibrated_cal_calibration_num_bins']

                    optimization_results['uncalibrated_test_ece'] = dataset_result['uncalibrated_test_ece']
                    optimization_results['uncalibrated_test_mce'] = dataset_result['uncalibrated_test_mce']
                    optimization_results['uncalibrated_test_conf_ece'] = dataset_result['uncalibrated_test_conf_ece']
                    optimization_results['uncalibrated_test_brier_score'] = dataset_result[
                        'uncalibrated_test_brier_score']
                    optimization_results['uncalibrated_test_calibration_curve_mean_predicted_probs'] = dataset_result[
                        'uncalibrated_test_calibration_curve_mean_predicted_probs']
                    optimization_results['uncalibrated_test_calibration_curve_true_probs'] = dataset_result[
                        'uncalibrated_test_calibration_curve_true_probs']
                    optimization_results['uncalibrated_test_calibration_num_bins'] = dataset_result[
                        'uncalibrated_test_calibration_num_bins']

                    # Flatten the results dictionary for CSV writing
                    flat_results = {}
                    for key, value in optimization_results.items():
                        if isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                flat_results[f"{key}_{sub_key}"] = sub_value
                        else:
                            flat_results[key] = value

                    # Determine fieldnames on first iteration
                    if fieldnames is None:
                        # Define the desired order of columns
                        fixed_columns = ['dataset_name', 'model_name', 'Total_Iterations', 'metric_used']
                        other_columns = set().union(*(result.keys() for result in [flat_results])) - set(fixed_columns)
                        fieldnames = fixed_columns + sorted(other_columns)

                        # Create CSV writer with determined fieldnames
                        writer = csv.DictWriter(f, fieldnames=fieldnames)

                        # Write header if needed
                        if write_header:
                            writer.writeheader()
                            write_header = False

                    # Write the current result to CSV
                    writer.writerow(flat_results)

                    # Flush to ensure writing to disk
                    f.flush()

    logging.info(f"Results for metric {current_metric} saved incrementally to {output_filename}")

logging.info("Completed running all metrics!")

success = combine_smartcal_csvs()
sys.exit(0 if success else 1) 