import pandas as pd
import logging
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
import os

from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment_V2
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from smartcal.meta_features_extraction.meta_features_extraction import MetaFeaturesExtractor
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.meta_model.meta_model import MetaModel

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
 
def get_calibration_and_performance_data(db, table, dataset_name):
    """Extract calibration set data and performance metrics"""
    try:
        # Query all relevant data including performance metrics for each calibrator
        calibration_data = (
            db.query(
                table.dataset_name,
                table.classification_type,
                table.classification_model,
                table.problem_type,
                table.ground_truth_cal_set,
                table.uncalibrated_probs_cal_set,
                table.calibration_algorithm,
                table.calibration_metric,
                table.calibrated_test_ece,
                table.calibrated_test_loss,
                table.calibrated_test_mce,
                table.calibrated_test_brier_score,
                table.calibrated_cal_conf_ece
            )
            .filter(
                table.dataset_name == dataset_name,
                table.status == Experiment_Status_Enum.COMPLETED.value
            )
            .all()
        )

        # Group data by dataset and model
        results = {}
        for data in calibration_data:
            key = (data.dataset_name, data.classification_model)
            
            if key not in results:
                results[key] = {
                    "dataset_name": data.dataset_name,
                    "dataset_type": data.classification_type,
                    "problem_type":data.problem_type,
                    "classification_model": data.classification_model,
                    "calibration_set_true_labels": data.ground_truth_cal_set,
                    "calibration_set_predicted_labels":data.uncalibrated_probs_cal_set,
                    "calibrator_performances": {}
                }
            
            # Store all performance metrics for each calibrator
            if data.calibration_algorithm not in results[key]["calibrator_performances"]:
                results[key]["calibrator_performances"][data.calibration_algorithm] = {}

            # Extract only the first value from conf_ece (where threshold = 0.2)
            conf_ece = data.calibrated_cal_conf_ece[0] if isinstance(data.calibrated_cal_conf_ece,
                                                                     list) else data.calibrated_cal_conf_ece

            # Store metrics by calibration_metric
            results[key]["calibrator_performances"][data.calibration_algorithm][data.calibration_metric] = {
                "ece": data.calibrated_test_ece,
                "loss": data.calibrated_test_loss,
                "mce": data.calibrated_test_mce,
                "brier_score": data.calibrated_test_brier_score,
                "conf_ece": conf_ece
            }
        return list(results.values())

    except Exception as e:
        logging.error(f"Error fetching data for dataset {dataset_name}: {e}")
        return []


def select_best_calibrator(predicted_calibrators, calibrator_performances, metric_name):
    """Select the best performing calibrator from the predicted set for a specific metric"""
    best_score = float('inf')
    best_cal = None

    performance_mapping = {
        "ECE": "ece",
        "MCE": "mce",
        "ConfECE": "conf_ece",
        "log_loss": "loss",
        "brier_score": "brier_score"
    }

    metric_field = performance_mapping.get(metric_name)
    if not metric_field:
        raise ValueError(f"Unknown metric: {metric_name}")

    for calibrator in predicted_calibrators:
        if calibrator in calibrator_performances:
            metrics_by_cal_metric = calibrator_performances[calibrator]
            # Find the metrics for this calibrator where calibration_metric matches our target
            for cal_metric, metrics in metrics_by_cal_metric.items():
                if cal_metric == metric_name and metric_field in metrics:
                    if metrics[metric_field] < best_score:
                        best_score = metrics[metric_field]
                        best_cal = calibrator

    return best_cal, best_score


def get_best_calibrator_from_db(calibrator_performances, metric_name):
    """Find the calibrator with the best performance for a specific metric from the database"""
    best_score = float('inf')
    best_cal = None

    performance_mapping = {
        "ECE": "ece",
        "MCE": "mce",
        "ConfECE": "conf_ece",
        "log_loss": "loss",
        "brier_score": "brier_score"
    }

    metric_field = performance_mapping.get(metric_name)
    if not metric_field:
        raise ValueError(f"Unknown metric: {metric_name}")

    for calibrator, metrics_by_cal_metric in calibrator_performances.items():
        for cal_metric, metrics in metrics_by_cal_metric.items():
            if cal_metric == metric_name and metric_field in metrics:
                if metrics[metric_field] < best_score:
                    best_score = metrics[metric_field]
                    best_cal = calibrator

    return best_cal, best_score


def generate_and_save_results(n=config_manager.k_recommendations, db_table=BenchmarkingExperiment_V2):
    """Generate results for different N values and save to CSV"""
    results_data = []
    with SessionLocal() as db:
        datasets = fetch_completed_datasets(db, db_table)
        print(f"Processing {len(datasets)} completed datasets: {datasets}")

        meta_model_metrics = config_manager.supported_metrics
        
        # Define all metrics we want to evaluate
        metrics = [
            ("ECE", "ece"),
            ("MCE", "mce"),
            ("ConfECE", "conf_ece"),
            ("log_loss", "loss"),
            ("brier_score", "brier_score")
        ]
        
        # Create 'results' directory if it doesn't exist
        if not os.path.exists('experiments/meta_model_eval/Results'):
            os.makedirs('experiments/meta_model_eval/Results')

        # Create/check if the results file exists
        csv_filename = os.path.join('experiments/meta_model_eval/Results', f"meta_model_results.csv")
        
        # Run for each meta model metric
        for meta_model_metric in meta_model_metrics:
            print(f"\nProcessing with meta model metric: {meta_model_metric}")
            
            # Initialize the meta model with the current metric
            meta_model = MetaModel(
                top_n=config_manager.meta_model_k,
                metric=meta_model_metric.lower(),
            )
            
            results_data = []
            
            for dataset_name in datasets:
                print(f"\nProcessing dataset: {dataset_name}")
                dataset_results = get_calibration_and_performance_data(db, db_table, dataset_name)

                for data in dataset_results:
                    y_true = data["calibration_set_true_labels"]
                    y_pred = data["calibration_set_predicted_labels"]
                    print(y_pred[0:5])
                    extractor = MetaFeaturesExtractor()
                    meta_features = extractor.process_features(y_true, y_pred)
                    print(f"Meta features extracted: {len(meta_features)} features")

                    # Get meta-model predictions (list of tuples)
                    predicted_calibrators_with_scores = meta_model.predict_best_model(meta_features)
                    print(f"Predicted calibrators: {predicted_calibrators_with_scores}")

                    # Extract calibrator names and confidence scores
                    predicted_calibrators = [calibrator for calibrator, _ in predicted_calibrators_with_scores]
                    confidence_scores = [score for _, score in predicted_calibrators_with_scores]

                    # Generate results for each N value using range(1, n + 1)
                    for N in range(1, n + 1):
                        top_n_calibrators = predicted_calibrators[:N]
                        top_n_confidences = confidence_scores[:N]

                        # Create a result row for each metric
                        for metric_name, _ in metrics:
                            # Select best performing calibrator from the top N predictions
                            best_cal, best_score = select_best_calibrator(
                                top_n_calibrators,
                                data["calibrator_performances"],
                                metric_name
                            )

                            # Get the best possible calibrator from DB for this metric
                            best_cal_db, best_score_db = get_best_calibrator_from_db(
                                data["calibrator_performances"],
                                metric_name
                            )

                            # Check if the chosen calibrator is the best according to the database
                            is_best_calibrator = best_cal == best_cal_db if (best_cal and best_cal_db) else False

                            results_data.append({
                                "Dataset_Name": data["dataset_name"],
                                "Dataset_Type": data["dataset_type"],
                                "Problem_Type": data["problem_type"],
                                "Model_Type": data["classification_model"],
                                "N": N,
                                "Predicted_Calibrators": ",".join(top_n_calibrators),
                                "Evaluation_Metric": metric_name,
                                "Best_Calibrator": best_cal,
                                "Best_Performance_Value": best_score,
                                "Best_Possible_Performance": best_score_db,
                                "Meta_Model_Confidence": top_n_confidences,
                                "Meta_Model_Version": config_manager.meta_model_version,
                                "Is_Best_Calibrator": is_best_calibrator,
                                "Meta_Model_Metric": meta_model_metric
                            })
                
            # Add to the combined results
            results_data.extend(results_data)
            print(f"Completed processing meta model metric: {meta_model_metric}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame(results_data)
    df.to_csv(csv_filename, index=False)

    print(f"\nResults saved to {csv_filename}")
    return df

if __name__ == "__main__":
    results_df = generate_and_save_results(n=config_manager.k_recommendations, db_table=BenchmarkingExperiment_V2)
    print("\nGenerated Results Summary:")
    print(results_df.groupby(['Dataset_Name', 'Evaluation_Metric', 'Meta_Model_Metric']).size().unstack())