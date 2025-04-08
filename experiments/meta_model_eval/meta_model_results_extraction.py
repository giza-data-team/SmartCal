import pandas as pd
import logging
from sqlalchemy.orm import Session, aliased
from sqlalchemy import func
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment_V2
from Package.src.SmartCal.meta_features_extraction.meta_features_extraction import MetaFeaturesExtractor
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from Package.src.SmartCal.meta_model.meta_model import MetaModel

config_manager = ConfigurationManager()

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
                table.calibrated_test_ece,
                table.calibrated_test_brier_score,
                table.calibrated_test_loss,
                table.calibration_algorithm
            )
            .filter(
                table.dataset_name == dataset_name,
                table.status == "completed"
            )
            .all()
        )

        # Group data by dataset and model
        results = {}
        for data in calibration_data:
            key = (data.dataset_name, data.classification_model)
            
            if key not in results:
                # predicted_classes = np.argmax(data.uncalibrated_probs_cal_set, axis=1).tolist()
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
            results[key]["calibrator_performances"][data.calibration_algorithm] = {
                "ece": data.calibrated_test_ece,
                "brier_score": data.calibrated_test_brier_score,
                "loss": data.calibrated_test_loss
            }
        return list(results.values())

    except Exception as e:
        logging.error(f"Error fetching data for dataset {dataset_name}: {e}")
        return []
   
def select_best_calibrator(predicted_calibrators, calibrator_performances):
    """Select the best performing calibrator from the predicted set using combined metrics"""
    best_score_ece = float('inf')
    best_score_brier_score = float('inf')
    best_score_loss = float('inf')
    best_cal_ece = None
    best_cal_brier_score = None
    best_cal_loss = None

    for calibrator in predicted_calibrators:
        if calibrator in calibrator_performances:
            metrics = calibrator_performances[calibrator]
            
            if metrics["ece"] < best_score_ece:
                best_score_ece = metrics["ece"]
                best_cal_ece = calibrator

            if metrics["brier_score"] < best_score_brier_score:
                best_score_brier_score = metrics["brier_score"]
                best_cal_brier_score = calibrator

            if metrics["loss"] < best_score_loss:
                best_score_loss = metrics["loss"]
                best_cal_loss = calibrator
    return (
        best_cal_ece, best_score_ece,
        best_cal_brier_score, best_score_brier_score,
        best_cal_loss, best_score_loss
    )

def get_best_calibrator_from_db(calibrator_performances):
    """Find the calibrator with the best performance for each metric from the database"""
    best_score_ece = float('inf')
    best_score_brier = float('inf')
    best_score_loss = float('inf')
    best_cal_ece = None
    best_cal_brier = None
    best_cal_loss = None
    
    for calibrator, metrics in calibrator_performances.items():
        # Track best ECE
        if metrics["ece"] < best_score_ece:
            best_score_ece = metrics["ece"]
            best_cal_ece = calibrator
            
        # Track best Brier Score
        if metrics["brier_score"] < best_score_brier:
            best_score_brier = metrics["brier_score"]
            best_cal_brier = calibrator
            
        # Track best Loss
        if metrics["loss"] < best_score_loss:
            best_score_loss = metrics["loss"]
            best_cal_loss = calibrator
    
    return (
        best_cal_ece, best_score_ece,
        best_cal_brier, best_score_brier,
        best_cal_loss, best_score_loss
    )

def generate_and_save_results(N_values=[1, 2, 3, 4, 5], db_table=BenchmarkingExperiment_V2):
    """Generate results for different N values and save to CSV"""
    results_data = []
    with SessionLocal() as db:
        datasets = fetch_completed_datasets(db, db_table)
        print(f"Processing {len(datasets)} completed datasets: {datasets}")

        meta_model = MetaModel(
                        top_n=5,  # or prob_threshold=0.9, etc.
                        model_path=config_manager.meta_model_path,
                        ordinal_encoder_path=config_manager.meta_ordinal_encoder_path,
                        label_encoder_path=config_manager.meta_label_encoder_path,
                        scaler_path=config_manager.meta_scaler_path
        )
        
        for dataset_name in datasets:
            print(f"\nProcessing dataset: {dataset_name}")
            dataset_results = get_calibration_and_performance_data(db,db_table, dataset_name)
            
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
                
                # Find the best calibrator from the database
                (best_calibrator_db_ece, best_ece_db, 
                best_calibrator_db_brier, best_brier_db, 
                best_calibrator_db_loss, best_loss_db) = get_best_calibrator_from_db(data["calibrator_performances"])
                # Generate results for each N value
                for N in N_values:
                    top_n_calibrators = predicted_calibrators[:N]
                    top_n_confidences = confidence_scores[:N]
                    
                    # Select best performing calibrator from the top N predictions
                    (
                        best_cal_ece, best_score_ece,
                        best_cal_brier_score, best_score_brier_score,
                        best_cal_loss, best_score_loss
                    ) = select_best_calibrator(
                        top_n_calibrators, 
                        data["calibrator_performances"]
                    )
                    
                    # Check if the chosen calibrator is the best according to the database
                    is_best_calibrator_ece = best_cal_ece == best_calibrator_db_ece
                    is_best_calibrator_brier = best_cal_brier_score == best_calibrator_db_brier
                    is_best_calibrator_loss = best_cal_loss == best_calibrator_db_loss
                    
                    # Create separate rows for each metric
                    metrics = [
                        {
                            "Dataset_Name": data["dataset_name"],
                            "Dataset_Type": data["dataset_type"],
                            "Problem_Type": data["problem_type"],
                            "Model_Type": data["classification_model"],
                            "N": N,
                            "Predicted_Calibrators": ",".join(top_n_calibrators),
                            "Evaluation_Metric": "ECE",
                            "Best_Calibrator": best_cal_ece,
                            "Best_Performance_Value": best_score_ece,
                            "Meta_Model_Confidence": top_n_confidences,
                            "Meta_Model_Version": config_manager.meta_model_version,
                            "Is_Best_Calibrator": is_best_calibrator_ece
                        },
                        {
                            "Dataset_Name": data["dataset_name"],
                            "Dataset_Type": data["dataset_type"],
                            "Problem_Type": data["problem_type"],
                            "Model_Type": data["classification_model"],
                            "N": N,
                            "Predicted_Calibrators": ",".join(top_n_calibrators),
                            "Evaluation_Metric": "Loss",
                            "Best_Calibrator": best_cal_loss,
                            "Best_Performance_Value": best_score_loss,
                            "Meta_Model_Confidence": top_n_confidences,
                            "Meta_Model_Version": config_manager.meta_model_version,
                            "Is_Best_Calibrator": is_best_calibrator_loss
                        },
                        {
                            "Dataset_Name": data["dataset_name"],
                            "Dataset_Type": data["dataset_type"],
                            "Problem_Type": data["problem_type"],
                            "Model_Type": data["classification_model"],
                            "N": N,
                            "Predicted_Calibrators": ",".join(top_n_calibrators),
                            "Evaluation_Metric": "Brier Score",
                            "Best_Calibrator": best_cal_brier_score,
                            "Best_Performance_Value": best_score_brier_score,
                            "Meta_Model_Confidence": top_n_confidences,
                            "Meta_Model_Version": config_manager.meta_model_version,
                            "Is_Best_Calibrator": is_best_calibrator_brier
                        }
                    ]
                    results_data.extend(metrics)
    
    # Create 'results' directory if it doesn't exist
    if not os.path.exists('experiments/meta_model_eval/Results'):
        os.makedirs('experiments/meta_model_eval/Results')

    # Create DataFrame and save to CSV in the 'results' folder
    df = pd.DataFrame(results_data)
    csv_filename = os.path.join('experiments/meta_model_eval/Results', f"meta_model_results.csv")
    df.to_csv(csv_filename, index=False)

    print(f"\nResults saved to {csv_filename}")
    return df

if __name__ == "__main__":
    results_df = generate_and_save_results(db_table=BenchmarkingExperiment_V2)
    print("\nGenerated Results Summary:")
    print(results_df.groupby(['Dataset_Name', 'Evaluation_Metric']).size().unstack())