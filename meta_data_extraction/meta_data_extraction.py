import logging

#logging.basicConfig(level=logging.WARNING) # this line makes you see warnings or hiher only in the terminal

from sqlalchemy.orm import Session
from sqlalchemy import func

from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import KnowledgeBaseExperiment_V2
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from smartcal.meta_features_extraction.meta_features_extraction import MetaFeaturesExtractor
from smartcal.utils.timer import time_operation


# Load configuration manager
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
    
def get_best_calibration_for_dataset(db: Session, table, dataset_name: str):
    """Fetch the best calibration details for each classification model in a given dataset,
    ensuring all calibrators are completed before selection.
    Only selects cases with a single best calibrator (no ties)."""
    try:
        # Create a subquery to identify completed calibrators
        subquery_calibrators = (
            db.query(
                table.dataset_name,
                table.classification_model,
                table.calibration_algorithm,
                table.calibrated_test_ece,
                func.rank().over(
                    partition_by=[table.dataset_name, table.classification_model],
                    order_by=table.calibrated_test_ece
                ).label("rank"),
                func.count().over(
                    partition_by=[table.dataset_name, table.classification_model, table.calibrated_test_ece]
                ).label("ece_tie_count")
            )
            .filter(
                table.dataset_name == dataset_name,
                table.status == Experiment_Status_Enum.COMPLETED.value
            )
            .subquery()
        )

        # Count how many calibration algorithms have ECE = 0 per dataset_name and classification_model
        subquery_zero_ece = (
            db.query(
                subquery_calibrators.c.dataset_name,
                subquery_calibrators.c.classification_model,
                func.count().label("zero_ece_algorithms")
            )
            .filter(subquery_calibrators.c.calibrated_test_ece == 0)
            .group_by(
                subquery_calibrators.c.dataset_name,
                subquery_calibrators.c.classification_model
            )
            .subquery()
        )

        # Select only the best calibrators with no ties and special handling for ECE = 0
        best_calibrations = (
            db.query(
                table.dataset_name,
                table.classification_model,
                table.calibration_algorithm,
                table.calibrated_test_ece,
                table.ground_truth_test_set,
                table.calibrated_probs_test_set
            )
            .join(
                subquery_calibrators,
                (table.dataset_name == subquery_calibrators.c.dataset_name) &
                (table.classification_model == subquery_calibrators.c.classification_model) &
                (table.calibration_algorithm == subquery_calibrators.c.calibration_algorithm)
            )
            .outerjoin(
                subquery_zero_ece,
                (table.dataset_name == subquery_zero_ece.c.dataset_name) &
                (table.classification_model == subquery_zero_ece.c.classification_model)
            )
            .filter(
                table.dataset_name == dataset_name,
                subquery_calibrators.c.rank == 1,
                subquery_calibrators.c.ece_tie_count == 1,  # Only cases with single best algorithm
                (subquery_zero_ece.c.zero_ece_algorithms.is_(None) |
                 (subquery_zero_ece.c.zero_ece_algorithms <= 1))  # Handle multiple algorithms with ECE=0
            )
            .all()
        )

        results = {}
        for experiment in best_calibrations:
            key = (experiment.dataset_name, experiment.classification_model)

            if key not in results:
                results[key] = {
                    "dataset_name": experiment.dataset_name,
                    "classification_model": experiment.classification_model,
                    "best_calibrators": [experiment.calibration_algorithm],
                    "min_calibrated_ece": experiment.calibrated_test_ece,
                    "y_true": experiment.ground_truth_test_set,
                    "predicted_probs": experiment.calibrated_probs_test_set,
                    "minimum_ece": experiment.calibrated_test_ece
                }

        return list(results.values())

    except Exception as e:
        logging.error(f"Error fetching best calibrators for dataset {dataset_name}: {e}")
        return []
    
@time_operation
def process_meta_features(table, path=config_manager.meta_data_file):
    """Process meta features for completed datasets, ensuring only datasets with a single best calibrator are processed."""
    config_manager = ConfigurationManager()
    meta_features_extractor = MetaFeaturesExtractor()

    # Use a session context manager to ensure proper handling of the database session
    with SessionLocal() as db:
        try:
            all_datasets = fetch_completed_datasets(db, table)
        except Exception as e:
            logging.error(f"Error fetching datasets: {e}")
            return

        for dataset in all_datasets:
            try:
                best_calibrations = get_best_calibration_for_dataset(db, table, dataset)

                if not best_calibrations:
                    logging.info(f"Skipping dataset '{dataset}': No valid best calibrator found.")
                    continue

                for calibration in best_calibrations:
                    dataset_name = calibration["dataset_name"]
                    model_name = calibration["classification_model"]
                    y_true = calibration["y_true"]
                    y_pred_prob = calibration["predicted_probs"]
                    best_cal = calibration["best_calibrators"][0]  # Extract the single best calibrator
                    min_ece = calibration["minimum_ece"]

                    # Validate y_true and y_pred_prob
                    if not y_true or not y_pred_prob:
                        logging.warning(f"Skipping {dataset_name} - {model_name}: Missing ground truth or predicted probabilities.")
                        continue

                    # Processing meta-features
                    logging.info(f"Processing: Dataset='{dataset_name}', Model='{model_name}', Best Calibrator='{best_cal}'")
                    
                    try:
                        #meta_features_extractor.set_dataset_name(dataset_name)
                        #meta_features_extractor.set_model_name(model_name)
                        meta_features_extractor.process_features(y_true, y_pred_prob)
                        meta_features_extractor.set_best_cal(best_cal)
                        #meta_features_extractor.set_min_ece(min_ece)

                        # Save after processing each dataset-model pair
                        meta_features_extractor.save_to_csv(path)

                    except Exception as e:
                        logging.error(f"Error processing features for {dataset_name} - {model_name} - {best_cal}: {e}")
                        continue  # Skip this dataset-model pair and move to the next

            except Exception as e:
                logging.error(f"Unexpected error while processing dataset '{dataset}': {e}")
                continue  # Continue to the next dataset

    logging.info("Meta feature extraction completed successfully.")

if __name__ == '__main__':
    process_meta_features(table=KnowledgeBaseExperiment_V2)