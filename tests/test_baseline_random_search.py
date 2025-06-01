import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os

from experiments.end_to_end_eval.baseline_random_search_script import (
    get_n_random_calibration_combinations,
    apply_calibration_with_cv,
    save_best_result,
    process_random_baseline
)


class TestBaselineRandomSearch(unittest.TestCase):

    def test_get_n_random_calibration_combinations(self):
        combinations = [{'param': i} for i in range(5)]
        selected = get_n_random_calibration_combinations(combinations, n=3, seed=42)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(item in combinations for item in selected))

    @patch("experiments.end_to_end_eval.baseline_random_search_script.StratifiedKFold.split")
    def test_apply_calibration_with_cv(self, mock_split):
        # Setup mock for StratifiedKFold
        mock_split.return_value = [(np.array([0]), np.array([1]))]

        # Use a valid calibration algorithm name from your Enum (e.g., TEMPERATURESCALING)
        calibrator_dict = {
            "Calibration_Algorithm": "TEMPERATURESCALING",
            "lr_tempscaling": 0.1,
            "initial_T": 1.0,
            "max_iter_tempscaling": 300
        }

        uncalibrated_probs_valid = np.array([[0.7, 0.3], [0.4, 0.6]])
        y_valid = np.array([0, 1])
        uncalibrated_probs_test = np.array([[0.5, 0.5], [0.2, 0.8]])

        # Call function
        test_probs, val_probs = apply_calibration_with_cv(
            calibrator_dict,
            uncalibrated_probs_valid,
            y_valid,
            uncalibrated_probs_test,
            n_splits=2
        )

        self.assertEqual(test_probs.shape, uncalibrated_probs_test.shape)
        self.assertEqual(val_probs.shape, uncalibrated_probs_valid.shape)

    @patch("experiments.end_to_end_eval.baseline_random_search_script.save_to_csv_incrementally")
    def test_save_best_result(self, mock_save_to_csv):
        dataset_name = "DatasetX"
        classification_model = "ModelY"
        output_path = "results/mock_output.csv"  # Provide a real subdirectory
        best_config = {
            "Calibration_Algorithm": "TEMPERATURESCALING",
            "lr_tempscaling": 0.1,
            "initial_T": 1.0,
            "max_iter_tempscaling": 300
        }

        classification_metrics_cal = {
            "loss": 0.1,
            "accuracy": 0.9,
            "f1_macro": 0.88,
            "f1_micro": 0.9,
            "f1_weighted": 0.89,
            "recall_macro": 0.87,
            "recall_micro": 0.91,
            "recall_weighted": 0.9,
            "precision_macro": 0.89,
            "precision_micro": 0.9,
            "precision_weighted": 0.91
        }
        cal_metrics_cal = {
            "ece": 0.03,
            "mce": 0.04,
            "conf_ece": 0.02,
            "brier_score": 0.05
        }

        save_best_result(
            dataset_name, classification_model, 10, best_config,
            0.15, 0.88, 0.06, 0.07, 0.05,
            0.84, 0.85, 0.86,
            0.82, 0.83, 0.84,
            0.8, 0.81, 0.82,
            0.09,
            classification_metrics_cal,
            cal_metrics_cal,
            classification_metrics_cal,
            cal_metrics_cal,
            output_path
        )

        mock_save_to_csv.assert_called_once()

    @patch("experiments.end_to_end_eval.baseline_random_search_script.save_best_result")
    @patch("experiments.end_to_end_eval.baseline_random_search_script.compute_classification_metrics")
    @patch("experiments.end_to_end_eval.baseline_random_search_script.compute_calibration_metrics")
    @patch("experiments.end_to_end_eval.baseline_random_search_script.apply_calibration_with_cv")
    @patch("experiments.end_to_end_eval.baseline_random_search_script.extract_uncalibrated_data")
    @patch("experiments.end_to_end_eval.baseline_random_search_script.fetch_completed_datasets")
    @patch("os.makedirs")
    def test_process_random_baseline(
        self,
        mock_makedirs,
        mock_fetch_datasets,
        mock_extract_data,
        mock_apply_cal,
        mock_cal_metrics,
        mock_clf_metrics,
        mock_save_best
    ):
        # Setup mocks
        mock_fetch_datasets.return_value = ["mock_dataset"]
        mock_extract_data.return_value = [{
            "classification_model": "MockModel",
            "uncalibrated_probs_cal_set": np.array([[0.6, 0.4], [0.3, 0.7]]),
            "uncalibrated_probs_test_set": np.array([[0.5, 0.5], [0.2, 0.8]]),
            "ground_truth_cal_set": np.array([0, 1]),
            "ground_truth_test_set": np.array([0, 1]),
            "uncalibrated_train_loss": 0.1,
            "uncalibrated_train_accuracy": 0.9,
            "uncalibrated_train_ece": 0.05,
            "uncalibrated_train_mce": 0.06,
            "uncalibrated_train_conf_ece": 0.04,
            "uncalibrated_train_f1_score_macro": 0.85,
            "uncalibrated_train_f1_score_micro": 0.87,
            "uncalibrated_train_f1_score_weighted": 0.88,
            "uncalibrated_train_recall_macro": 0.83,
            "uncalibrated_train_recall_micro": 0.84,
            "uncalibrated_train_recall_weighted": 0.85,
            "uncalibrated_train_precision_macro": 0.86,
            "uncalibrated_train_precision_micro": 0.87,
            "uncalibrated_train_precision_weighted": 0.88,
            "uncalibrated_train_brier_score": 0.07
        }]

        mock_apply_cal.return_value = (
            np.array([[0.6, 0.4], [0.4, 0.6]]),
            np.array([[0.65, 0.35], [0.3, 0.7]])
        )

        mock_cal_metrics.return_value = {
            "ece": 0.02,
            "mce": 0.03,
            "conf_ece": 0.01,
            "brier_score": 0.04
        }

        mock_clf_metrics.return_value = {
            "loss": 0.1,
            "accuracy": 0.9,
            "f1_macro": 0.89,
            "f1_micro": 0.91,
            "f1_weighted": 0.9,
            "recall_macro": 0.88,
            "recall_micro": 0.92,
            "recall_weighted": 0.91,
            "precision_macro": 0.87,
            "precision_micro": 0.9,
            "precision_weighted": 0.89
        }

        db_mock = MagicMock()
        output_path = "results/mock_output.csv"
        process_random_baseline(db_mock, MagicMock(), output_path)

        self.assertTrue(mock_fetch_datasets.called)
        self.assertTrue(mock_extract_data.called)
        self.assertTrue(mock_apply_cal.called)
        self.assertTrue(mock_save_best.called)
        mock_makedirs.assert_called_once_with(os.path.dirname(output_path), exist_ok=True)


if __name__ == "__main__":
    unittest.main()
