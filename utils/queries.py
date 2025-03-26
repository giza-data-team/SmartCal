from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
import logging
import numpy as np


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


def get_best_calibration_for_dataset(db: Session, table, dataset_name: str):
    """Fetch the best calibration details for each classification model in a given dataset,
    ensuring all calibrators are completed before selection, and excluding IMAX entirely.
    Only selects cases with a single best calibrator (no ties)."""
    try:
        # Create a subquery to identify completed calibrators, excluding IMAX
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
                table.status == "completed",
                table.calibration_algorithm != "IMAX"
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
                table.ground_truth_test_set
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
                "uncalibrated_probs_valid": np.array(experiment.uncalibrated_probs_cal_set),
                "y_valid": np.array(experiment.ground_truth_cal_set),
                "uncalibrated_probs_test": np.array(experiment.uncalibrated_probs_test_set),
                "y_test": np.array(experiment.ground_truth_test_set)
            })

        return extracted_data

    except Exception as e:
        logging.error(f"Error extracting uncalibrated data for dataset {dataset_name}: {e}")
        db.rollback()  # Recover from error
        return []
