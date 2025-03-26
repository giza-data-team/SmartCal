import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hashlib
import pandas as pd
import logging
from sqlalchemy.orm import Session
from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment, KnowledgeBaseExperiment, BenchmarkingExperiment_V2, \
    KnowledgeBaseExperiment_V2
from config.configuration_manager import ConfigurationManager

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
                model_class.calibrated_test_ece
            )
            .filter(model_class.status == "completed")
            .all()
        )

        return pd.DataFrame(data, columns=["dataset", "dataset_type", "problem_type", "model", "calibrator",
                                           "test_performance"])

    except Exception as e:
        logging.error(f"Error fetching calibration data from {model_class.__name__}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure


def select_best_calibrator(benchmarking_data, n=config_manager.n_calibrators):
    """Randomly samples N calibrators and selects the best one based on test performance (lower ECE is better)."""
    results = []

    for N in range(1, n + 1):
        for (dataset, model), group in benchmarking_data.groupby(['dataset', 'model']):
            seed = get_hashed_seed(dataset, model)
            sampled = group.sample(n=min(N, len(group)), random_state=seed)
            best_calibrator = sampled.loc[sampled['test_performance'].idxmin()]  # Select lowest ECE

            results.append((
                dataset,
                group['dataset_type'].iloc[0],
                group['problem_type'].iloc[0],
                model,
                N,
                list(sampled['calibrator']),
                best_calibrator['calibrator'],
                best_calibrator['test_performance']
            ))

    return results


def save_results_to_csv(results, filename=config_manager.baseline_results):
    """Saves the results to a CSV file."""
    df = pd.DataFrame(results, columns=[
        'Dataset_Name',
        'Dataset_Type',
        'Problem_Type',
        'Model_Type',
        'N',
        'Selected_N',
        'Final_Selected_Calibrator',
        'Final_Test_Performance'
    ])
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

    #For default:
    # main()