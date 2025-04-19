import hashlib
import pandas as pd
import logging
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
    Fetch dataset_name, classification_model, calibration_algorithm, and calibrated_test_ece from the database.

    Args:
        db (Session): SQLAlchemy database session
        model_class: SQLAlchemy model class to query from (default: BenchmarkingExperiment)

    Returns:
        pd.DataFrame: DataFrame containing the query results
    """
    try:
        data = (
            db.query(
                model_class.dataset_name,
                model_class.classification_type,
                model_class.problem_type,
                model_class.classification_model,
                model_class.calibration_algorithm,
                model_class.calibrated_test_ece,
                model_class.calibrated_test_loss,
                model_class.calibrated_test_brier_score
            )
            .filter(model_class.status ==  Experiment_Status_Enum.COMPLETED.value)
            .all()
        )

        return pd.DataFrame(data, columns=["dataset", "dataset_type", "problem_type", "model",
                                           "calibrator", "test_performance_ece",
                                           "test_performance_loss", "test_performance_brier_score"])

    except Exception as e:
        logging.error(f"Error fetching calibration data from {model_class.__name__}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure


def select_best_calibrator(benchmarking_data, n=config_manager.k_recommendations):
    """
    Randomly samples N calibrators and selects the best one based on test performance.
    Returns a list of rows with each metric reported separately.
    """
    results = []

    for N in range(1, n + 1):
        for (dataset, model), group in benchmarking_data.groupby(['dataset', 'model']):
            seed = get_hashed_seed(dataset, model)
            sampled = group.sample(n=min(N, len(group)), random_state=seed)

            if sampled.empty:
                continue  # Skip empty sampled data

            # Metrics to process
            metrics = [
                ("ECE", "test_performance_ece"),
                ("Loss", "test_performance_loss"),
                ("Brier Score", "test_performance_brier_score")
            ]

            for metric_name, column in metrics:
                # Sort the sampled data by the current metric
                sorted_sampled = sampled.sort_values(by=column)

                # Select the best calibrator
                best_calibrator = sorted_sampled.iloc[0]

                results.append({
                    'Dataset_Name': dataset,
                    'Dataset_Type': best_calibrator['dataset_type'],
                    'Problem_Type': best_calibrator['problem_type'],
                    'Model_Type': model,
                    'N': N,
                    'Selected_Calibrators': list(sampled['calibrator']),
                    'Evaluation_Metric': metric_name,
                    'Best_Calibrator': best_calibrator['calibrator'],
                    'Best_Performance_Value': best_calibrator[column]
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
