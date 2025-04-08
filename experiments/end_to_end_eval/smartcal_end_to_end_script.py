import csv
import os
import sys
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
import logging
import numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment_V2
from Package.src.SmartCal.bayesian_optimization.calibrators_bayesian_optimization import CalibrationOptimizer
from Package.src.SmartCal.meta_model.meta_model import MetaModel
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()

meta_model = MetaModel(
                top_n=5,  # or prob_threshold=0.9, etc.
                model_path=config_manager.meta_model_path,
                ordinal_encoder_path=config_manager.meta_ordinal_encoder_path,
                label_encoder_path=config_manager.meta_label_encoder_path,
                scaler_path=config_manager.meta_scaler_path
)

optimizer = CalibrationOptimizer(meta_model=meta_model)

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
                table.status == "completed"
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
                    ground_truth_cal = np.array(eval(data.ground_truth_cal_set)) if isinstance(data.ground_truth_cal_set, str) else np.array(data.ground_truth_cal_set)
                    uncal_probs_cal = np.array(eval(data.uncalibrated_probs_cal_set)) if isinstance(data.uncalibrated_probs_cal_set, str) else np.array(data.uncalibrated_probs_cal_set)
                    ground_truth_test = np.array(eval(data.ground_truth_test_set)) if isinstance(data.ground_truth_test_set, str) else np.array(data.ground_truth_test_set)
                    uncal_probs_test = np.array(eval(data.uncalibrated_probs_test_set)) if isinstance(data.uncalibrated_probs_test_set, str) else np.array(data.uncalibrated_probs_test_set)
                    
                    # Debugging: Print the probabilities for each dataset and classification model
                    # logging.info(f"Dataset: {data.dataset_name}, Model: {data.classification_model}")
                    # logging.info(f"Calibration Set Probabilities: {uncal_probs_cal[:5]}")  # Print first 5 probabilities
                    # logging.info(f"Test Set Probabilities: {uncal_probs_test[:5]}")  # Print first 5 probabilities
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
                    print(f"Error processing data for {key}: {e}")
                    continue

        return list(results.values())

    except Exception as e:
        logging.error(f"Error fetching data for dataset {dataset_name}: {e}", exc_info=True)
        return []
    
# Output folder and filename
output_folder = 'experiments/end_to_end_eval/Results'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
output_filename = os.path.join(output_folder, 'smartcal_results.csv')

# Iterations to run
iterations_list = [10, 30, 50]

# Flag to check if header needs to be written
write_header = not os.path.exists(output_filename)

# Open the CSV file in append mode
with open(output_filename, mode='a', newline='') as f, SessionLocal() as db:
    # Prepare CSV writer
    fieldnames = None
    writer = None

    # Fetch completed datasets
    datasets = fetch_completed_datasets(db, BenchmarkingExperiment_V2)
    
    # Iterate through different iteration counts
    for itr in iterations_list:
        print(f"Running optimization with {itr} iterations...")
        
        for dataset_name in datasets:
            results = fetch_cal_and_test_data(db, BenchmarkingExperiment_V2, dataset_name)
            
            for dataset_result in results:
                # Extract model name for clarity
                model_name = dataset_result['classification_model']
                
                # Run optimization for the current model
                optimization_results = optimizer.run_optimization(
                    dataset_result, total_iterations=itr
                )
                
                # Add dataset name and model name to the results
                optimization_results['dataset_name'] = dataset_name
                optimization_results['model_name'] = model_name
                optimization_results['Total_Iterations'] = itr
                    
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
                optimization_results['train_calibration_curve_mean_predicted_probs'] = dataset_result['train_calibration_curve_mean_predicted_probs']
                optimization_results['train_calibration_curve_true_probs'] = dataset_result['train_calibration_curve_true_probs']
                optimization_results['train_calibration_num_bins'] = dataset_result['train_calibration_num_bins']

                # Add the validation and test uncrossed metrics
                optimization_results['uncalibrated_cal_recall_micro'] = dataset_result['uncalibrated_cal_recall_micro']
                optimization_results['uncalibrated_cal_recall_macro'] = dataset_result['uncalibrated_cal_recall_macro']
                optimization_results['uncalibrated_cal_precision_micro'] = dataset_result['uncalibrated_cal_precision_micro']
                optimization_results['uncalibrated_cal_precision_macro'] = dataset_result['uncalibrated_cal_precision_macro']
                optimization_results['uncalibrated_cal_f1_score_micro'] = dataset_result['uncalibrated_cal_f1_score_micro']
                optimization_results['uncalibrated_cal_f1_score_macro'] = dataset_result['uncalibrated_cal_f1_score_macro']
                optimization_results['uncalibrated_cal_ece'] = dataset_result['uncalibrated_cal_ece']
                optimization_results['uncalibrated_cal_mce'] = dataset_result['uncalibrated_cal_mce']
                optimization_results['uncalibrated_cal_conf_ece'] = dataset_result['uncalibrated_cal_conf_ece']
                optimization_results['uncalibrated_cal_brier_score'] = dataset_result['uncalibrated_cal_brier_score']
                optimization_results['uncalibrated_cal_calibration_curve_mean_predicted_probs'] = dataset_result['uncalibrated_cal_calibration_curve_mean_predicted_probs']
                optimization_results['uncalibrated_cal_calibration_curve_true_probs'] = dataset_result['uncalibrated_cal_calibration_curve_true_probs']
                optimization_results['uncalibrated_cal_calibration_num_bins'] = dataset_result['uncalibrated_cal_calibration_num_bins']

                optimization_results['uncalibrated_test_ece'] = dataset_result['uncalibrated_test_ece']
                optimization_results['uncalibrated_test_mce'] = dataset_result['uncalibrated_test_mce']
                optimization_results['uncalibrated_test_conf_ece'] = dataset_result['uncalibrated_test_conf_ece']
                optimization_results['uncalibrated_test_brier_score'] = dataset_result['uncalibrated_test_brier_score']
                optimization_results['uncalibrated_test_calibration_curve_mean_predicted_probs'] = dataset_result['uncalibrated_test_calibration_curve_mean_predicted_probs']
                optimization_results['uncalibrated_test_calibration_curve_true_probs'] = dataset_result['uncalibrated_test_calibration_curve_true_probs']
                optimization_results['uncalibrated_test_calibration_num_bins'] = dataset_result['uncalibrated_test_calibration_num_bins']

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
                    fixed_columns = ['dataset_name', 'model_name', 'Total_Iterations']
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

print(f"Results saved incrementally to {output_filename}")