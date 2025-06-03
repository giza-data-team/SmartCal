# Configure logging
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
config_manager = ConfigurationManager()

import logging
logging.basicConfig(level=logging.WARNING, force=config_manager.logging) # this line makes you see warnings or higher only in the terminal

from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Numeric

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
    """
    Return only uniquely best calibration algorithms per model and metric.
    Supports array-type metrics (e.g., calibrated_test_conf_ece[]) by selecting the first element.
    Applies rounding on casted numeric values for PostgreSQL compatibility.
    Skips cases where there are ties at the minimum score (e.g., multiple 0.0 values).
    """
    try:
        calibration_metrics = [
            ("calibrated_test_ece", "ECE"),
            ("calibrated_test_mce", "MCE"),
            ("calibrated_test_conf_ece", "ConfECE"),
            ("calibrated_test_brier_score", "brier_score"),
            ("calibrated_test_loss", "log_loss")
        ]

        results = []

        for metric_column, metric_name in calibration_metrics:
            try:
                raw_metric = getattr(table, metric_column)
                metric_value = raw_metric[1] if metric_column == "calibrated_test_conf_ece" else raw_metric
                rounded_metric = func.round(cast(metric_value, Numeric), 10)

                # Subquery: Get min score per dataset+model+metric
                min_scores_subq = (
                    db.query(
                        table.dataset_name.label("ds"),
                        table.classification_model.label("model"),
                        table.calibration_metric.label("metric"),
                        func.round(func.min(cast(metric_value, Numeric)), 10).label("min_score")
                    )
                    .filter(
                        table.dataset_name == dataset_name,
                        table.calibration_metric == metric_name,
                        table.status == Experiment_Status_Enum.COMPLETED.value,
                        raw_metric.isnot(None)
                    )
                    .group_by(table.dataset_name, table.classification_model, table.calibration_metric)
                    .subquery()
                )

                # Subquery: Join to get all rows at the min score
                tied_subq = (
                    db.query(
                        table.id.label("exp_id"),
                        table.dataset_name,
                        table.classification_model,
                        table.calibration_algorithm,
                        table.calibration_metric,
                        rounded_metric.label("score"),
                        table.ground_truth_test_set.label("y_true"),
                        table.calibrated_probs_test_set.label("predicted_probs"),
                        func.count().over(
                            partition_by=[
                                table.dataset_name,
                                table.classification_model,
                                table.calibration_metric,
                                rounded_metric
                            ]
                        ).label("tie_count")
                    )
                    .join(min_scores_subq,
                          (table.dataset_name == min_scores_subq.c.ds) &
                          (table.classification_model == min_scores_subq.c.model) &
                          (table.calibration_metric == min_scores_subq.c.metric) &
                          (rounded_metric == min_scores_subq.c.min_score)
                          )
                    .filter(table.calibration_metric == metric_name)
                    .subquery()
                )

                # Final: Keep only unique best (tie_count == 1)
                final = (
                    db.query(
                        tied_subq.c.dataset_name,
                        tied_subq.c.classification_model,
                        tied_subq.c.calibration_algorithm,
                        tied_subq.c.calibration_metric,
                        tied_subq.c.score,
                        tied_subq.c.y_true,
                        tied_subq.c.predicted_probs
                    )
                    .filter(tied_subq.c.tie_count == 1)
                    .all()
                )

                for row in final:
                    results.append({
                        "dataset_name": row.dataset_name,
                        "classification_model": row.classification_model,
                        "calibration_metric": row.calibration_metric,
                        "best_calibrator": row.calibration_algorithm,
                        "score": row.score,
                        "y_true": row.y_true,
                        "predicted_probs": row.predicted_probs
                    })

            except Exception as e:
                logging.error(f"Error while processing metric '{metric_name}' for dataset '{dataset_name}': {e}")
                db.rollback()
                continue

        return results

    except Exception as e:
        logging.error(f"Error fetching best calibrators for dataset {dataset_name}: {e}")
        db.rollback()
        return []
    
@time_operation
def process_meta_features(table, path=config_manager.meta_data_file):
    """Process meta features for completed datasets, handling multiple calibration metrics."""
    meta_features_extractor = MetaFeaturesExtractor()

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
                    best_cal = calibration["best_calibrator"]
                    metric_name = calibration["calibration_metric"]
                    score = calibration["score"]

                    # Validate y_true and y_pred_prob
                    if not y_true or not y_pred_prob:
                        logging.warning(f"Skipping {dataset_name} - {model_name}: Missing ground truth or predicted probabilities.")
                        continue

                    logging.info(f"Processing: Dataset='{dataset_name}', Model='{model_name}', Metric='{metric_name}', Best Calibrator='{best_cal}'")

                    try:
                        meta_features_extractor.set_dataset_name(dataset_name)
                        meta_features_extractor.set_model_name(model_name)
                        meta_features_extractor.set_calibration_metric(metric_name)

                        # Process features
                        meta_features_extractor.process_features(y_true, y_pred_prob)
                        meta_features_extractor.set_best_cal(best_cal)

                        # Save after processing each dataset-model-metric
                        meta_features_extractor.save_to_csv(path)

                    except Exception as e:
                        logging.error(f"Error processing features for {dataset_name} - {model_name} - {metric_name} - {best_cal}: {e}")
                        continue  # Skip this dataset-model-metric and move to the next

            except Exception as e:
                logging.error(f"Unexpected error while processing dataset '{dataset}': {e}")
                continue  # Continue to the next dataset

    logging.info("Meta feature extraction completed successfully.")

if __name__ == '__main__':
    process_meta_features(table=KnowledgeBaseExperiment_V2)