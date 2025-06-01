import hashlib
import pandas as pd
import numpy as np
import logging
import os
from sqlalchemy.orm import Session

from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment, BenchmarkingExperiment_V2
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum

# Load configuration manager
config_manager = ConfigurationManager()


def get_hashed_seed(dataset, model):
    """Generate a consistent random seed from dataset and model."""
    return int(hashlib.sha256(f"{dataset}_{model}".encode()).hexdigest(), 16) % (10 ** 8)


def fetch_calibration_data(db: Session, model_class=BenchmarkingExperiment):
    """
    Fetch dataset_name, classification_model, calibration_algorithm, calibration_metric and test metrics.
    """
    try:
        data = (
            db.query(
                model_class.dataset_name,
                model_class.classification_type,
                model_class.problem_type,
                model_class.classification_model,
                model_class.calibration_algorithm,
                model_class.calibration_metric,
                model_class.calibrated_test_ece,
                model_class.calibrated_test_loss,
                model_class.calibrated_test_mce,
                model_class.calibrated_test_brier_score,
                model_class.calibrated_cal_conf_ece
            )
            .filter(model_class.status == Experiment_Status_Enum.COMPLETED.value)
            .all()
        )

        df = pd.DataFrame(data, columns=["dataset", "dataset_type", "problem_type", "model",
                                         "calibrator", "calibration_metric", "test_performance_ece",
                                         "test_performance_loss", "test_performance_mce",
                                         "test_performance_brier_score", "test_performance_conf_ece"])

        # Extract only the first value from conf_ece (where threshold = 0.2)
        df['test_performance_conf_ece'] = df['test_performance_conf_ece'].apply(
            lambda x: x[0] if isinstance(x, list) else x
        )

        return df

    except Exception as e:
        logging.error(f"Error fetching calibration data from {model_class.__name__}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure


def load_calibrator_distributions():
    """Load the calibrator distribution data from CSV."""
    dist_file = config_manager.all_class_distributions_file
    
    if os.path.exists(dist_file):
        return pd.read_csv(dist_file)
    else:
        logging.error(f"Calibrator distributions file not found at: {dist_file}")
        return pd.DataFrame()


def select_best_calibrator(benchmarking_data, n=config_manager.k_recommendations):
    """
    For each dataset/model/N, create one row for each metric type.
    Use metric-specific weighted calibrator selection based on distribution probabilities.
    For each metric, get the performance value from the row where calibration_metric matches that metric.
    """
    results = []
    
    # Load calibrator distributions
    dist_df = load_calibrator_distributions()
    if dist_df.empty:
        logging.error("Could not load calibrator distributions. Using uniform sampling instead.")
    
    # Define metrics and their corresponding performance columns
    metrics = [
        ("ECE", "test_performance_ece"),
        ("MCE", "test_performance_mce"),
        ("ConfECE", "test_performance_conf_ece"),
        ("log_loss", "test_performance_loss"),
        ("brier_score", "test_performance_brier_score")
    ]

    for N in range(1, n + 1):
        for (dataset, model), group in benchmarking_data.groupby(['dataset', 'model']):
            seed = get_hashed_seed(dataset, model)

            # Get sample row for metadata
            if group.empty:
                continue
            sample_row = group.iloc[0]
            
            # For each metric type, create a separate row with metric-specific sampling
            for metric_name, column_name in metrics:
                # Get the calibrator distribution for this metric
                if not dist_df.empty:
                    metric_dist = dist_df[dist_df['Metric'] == metric_name]
                else:
                    metric_dist = pd.DataFrame()
                    
                # Get available calibrators for this dataset/model
                available_calibrators = group['calibrator'].unique()
                
                if len(available_calibrators) <= N:
                    # If we have fewer unique calibrators than N, use all of them
                    selected_calibrators = list(available_calibrators)
                else:
                    # Sample N calibrators based on weighted distribution for this metric
                    np.random.seed(seed)
                    
                    if not metric_dist.empty:
                        # Filter distribution to only include available calibrators
                        valid_dist = metric_dist[metric_dist['Class'].isin(available_calibrators)]
                        
                        if not valid_dist.empty:
                            logging.info("Using weighted")
                            # Use weighted sampling based on the probabilities
                            calibrators = valid_dist['Class'].values
                            probabilities = valid_dist['Probabilities'].values

                            selected_calibrators = list(np.random.choice(calibrators, size=min(N, len(calibrators)),
                                                                         replace=False,p=probabilities))
                        else:
                            # Fallback to uniform sampling if no distribution data for available calibrators
                            selected_calibrators = list(np.random.choice(available_calibrators, N, replace=False))
                    else:
                        # Fallback to uniform sampling if no distribution data
                        selected_calibrators = list(np.random.choice(available_calibrators, N, replace=False))

                # Filter the group to only include the selected calibrators
                filtered_group = group[group['calibrator'].isin(selected_calibrators)]

                if filtered_group.empty:
                    continue

                # Get rows where calibration_metric matches current metric
                metric_rows = filtered_group[filtered_group['calibration_metric'] == metric_name]

                if metric_rows.empty:
                    # If no rows match this metric, use any row but set value to None
                    best_calibrator_name = selected_calibrators[0]
                    best_performance = None
                else:
                    # Find best calibrator for this metric
                    best_idx = metric_rows[column_name].idxmin()
                    best_calibrator = metric_rows.loc[best_idx]
                    best_calibrator_name = best_calibrator['calibrator']
                    best_performance = best_calibrator[column_name]

                results.append({
                    'Dataset_Name': dataset,
                    'Dataset_Type': sample_row['dataset_type'],
                    'Problem_Type': sample_row['problem_type'],
                    'Model_Type': model,
                    'N': N,
                    'Selected_Calibrators': selected_calibrators,
                    'Evaluation_Metric': metric_name,
                    'Best_Calibrator': best_calibrator_name,
                    'Best_Performance_Value': best_performance
                })

    return results


def save_results_to_csv(results, filename=config_manager.baseline_results):
    """Saves the results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def main(model_class=BenchmarkingExperiment):
    """
    Main function to run the calibration selection process.

    Args:
        model_class: SQLAlchemy model class to query from (default: BenchmarkingExperiment)
    """
    with SessionLocal() as db:
        data = fetch_calibration_data(db, model_class)

        if data.empty:
            print(f"No valid data fetched from {model_class.__name__}. Exiting.")
            return

        results = select_best_calibrator(data)
        save_results_to_csv(results)


if __name__ == "__main__":
    # You can specify a different model class here if needed
    main(BenchmarkingExperiment_V2)

    # For default:
    # main()