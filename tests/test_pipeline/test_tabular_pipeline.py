import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum

from pipeline.tabular_pipeline import TabularPipeline
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager

class TestTabularPipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock configuration
        self.config = {
            "dataset_path": "test_data.csv",
            "dataset_name": "test_dataset",
            "task_type": "classification",
            "logs": True,
            "experiment_type": "test_experiment",
            "combinations": {
                "RANDOM_FOREST": [
                    ("run_id_1", "PlattScaling", {"max_iter": [100, 200]}),
                    ("run_id_2", "IsotonicRegression", {})
                ]
            }
        }
        
        # Create pipeline instance with mocked ConfigurationManager
        with patch('pipeline.tabular_pipeline.ConfigurationManager') as mock_config:
            mock_config.return_value.random_seed = 42
            mock_config.return_value.n_bins = 10
            mock_config.return_value.conf_thresholds_list = [0.5, 0.7, 0.9]
            mock_config.return_value.config_tabular = "mock_path"
            mock_config.return_value.split_ratios = [0.7, 0.15, 0.15]
            self.pipeline = TabularPipeline(self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.config, self.config)
        self.assertIsNotNone(self.pipeline.logger)
        self.assertEqual(len(self.pipeline.classifiers), 0)
        self.assertEqual(len(self.pipeline.results), 0)
        self.assertEqual(len(self.pipeline.failed_models), 0)

    @patch('pipeline.tabular_pipeline.TabularSplitter')
    @patch('pipeline.tabular_pipeline.TabularPreprocessor')
    def test_load_preprocess_data(self, mock_preprocessor, mock_splitter):
        """Test data loading and preprocessing"""
        # Mock splitter
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_dataset.return_value = (
            self.sample_data.iloc[:60],
            self.sample_data.iloc[60:80],
            self.sample_data.iloc[80:]
        )
        mock_splitter.return_value = mock_splitter_instance

        # Mock preprocessor
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.fit_transform.return_value = (
            np.random.rand(60, 2),
            np.random.randint(0, 2, 60)
        )
        mock_preprocessor_instance.transform.side_effect = [
            (np.random.rand(20, 2), np.random.randint(0, 2, 20)),
            (np.random.rand(20, 2), np.random.randint(0, 2, 20))
        ]
        mock_preprocessor_instance.get_timing.return_value = {"preprocess_time": 0.2}
        mock_preprocessor.return_value = mock_preprocessor_instance

        # Test data loading
        with patch('pandas.read_csv', return_value=self.sample_data):
            self.pipeline.load_preprocess_data()

        # Assertions
        self.assertIsNotNone(self.pipeline.X_train)
        self.assertIsNotNone(self.pipeline.y_train)
        self.assertIsNotNone(self.pipeline.X_valid)
        self.assertIsNotNone(self.pipeline.y_valid)
        self.assertIsNotNone(self.pipeline.X_test)
        self.assertIsNotNone(self.pipeline.y_test)
        self.assertIsNotNone(self.pipeline.preprocessing_timing_info)

    @patch('pipeline.tabular_pipeline.ModelCache')
    def test_initialize_model(self, mock_model_cache):
        """Test model initialization"""
        # Mock model cache
        mock_model = Mock()
        mock_model_cache.get_model.return_value = mock_model

        self.pipeline.initialize_model()

        # Assertions
        self.assertEqual(len(self.pipeline.classifiers), 1)
        mock_model_cache.get_model.assert_called_once()
        self.assertNotIn("RANDOM_FOREST", self.pipeline.failed_models)

    def test_train_model(self):
        """Test model training"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.training_metrics = {"accuracy": 0.9}
        mock_classifier.training_time = 0.5
        mock_classifier.train = Mock()
        
        self.pipeline.classifiers = {"RANDOM_FOREST": mock_classifier}
        self.pipeline.X_train = np.random.rand(60, 2)
        self.pipeline.y_train = np.random.randint(0, 2, 60)

        self.pipeline.train_model()

        # Assertions
        mock_classifier.train.assert_called_once()
        self.assertIn("RANDOM_FOREST", self.pipeline.results)
        self.assertEqual(
            self.pipeline.results["RANDOM_FOREST"]["training_metrics"],
            {"accuracy": 0.9}
        )
        self.assertNotIn("RANDOM_FOREST", self.pipeline.failed_models)

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = np.random.randint(0, 2, 20)
        mock_classifier.predict_prob.return_value = np.random.rand(20, 2)
        mock_classifier.testing_metrics = {}
        mock_classifier.testing_time_predictprob = 0.1

        self.pipeline.classifiers = {"RANDOM_FOREST": mock_classifier}
        self.pipeline.X_test = np.random.rand(20, 2)
        self.pipeline.y_test = np.random.randint(0, 2, 20)
        self.pipeline.results = {"RANDOM_FOREST": {}}

        self.pipeline.evaluate_model()

        # Assertions
        self.assertIn("testing_metrics", self.pipeline.results["RANDOM_FOREST"])
        self.assertIn("predict_proba_time", self.pipeline.results["RANDOM_FOREST"])
        self.assertIn("probabilities", self.pipeline.results["RANDOM_FOREST"])
        self.assertNotIn("RANDOM_FOREST", self.pipeline.failed_models)

    @patch('pipeline.tabular_pipeline.tune_all_calibration')
    def test_calibrate_model(self, mock_tune_calibration):
        """Test model calibration"""
        # Setup mock classifier
        mock_classifier = Mock()
        mock_classifier.predict_prob.return_value = np.random.rand(20, 2)
        
        # Setup pipeline attributes
        self.pipeline.classifiers = {"RANDOM_FOREST": mock_classifier}
        self.pipeline.X_valid = np.random.rand(20, 2)
        self.pipeline.y_valid = np.random.randint(0, 2, 20)
        self.pipeline.X_test = np.random.rand(20, 2)
        self.pipeline.y_test = np.random.randint(0, 2, 20)
        self.pipeline.n_instances = 20
        self.pipeline.preprocessing_timing_info = {"preprocess_time": 0.2}

        # Setup results
        self.pipeline.results = {
            "RANDOM_FOREST": {
                "training_time": 0.1,
                "predict_proba_time": 0.1,
                "training_metrics": {"accuracy": 0.9},
                "testing_metrics": {"accuracy": 0.85},
                "probabilities": np.random.rand(20, 2)
            }
        }

        # Mock calibration results
        mock_tune_calibration.return_value = {
            "PlattScaling": {
                "best_hyperparams": {"max_iter": 100},
                "calibrated_metrics": {
                    "accuracy": 0.87,
                    "ece": 0.08
                }
            }
        }

        # Mock experiment manager if needed
        self.pipeline.experiment_manager = Mock()

        # Execute calibration
        self.pipeline.calibrate_model()

        # Assertions
        mock_classifier.predict_prob.assert_called()
        mock_tune_calibration.assert_called()
        
        # Verify experiment manager was called if it exists
        if hasattr(self.pipeline, 'experiment_manager'):
            self.pipeline.experiment_manager._save_results.assert_called()

        # Verify no models failed
        self.assertNotIn("RANDOM_FOREST", self.pipeline.failed_models)

    def test_save_failure_status(self):
        """Test failure status creation"""
        stage = "test_stage"
        error_message = "test error"
        
        failure_info = self.pipeline.save_failure_status(stage, error_message)
        
        # Assertions
        self.assertEqual(failure_info["status"], Experiment_Status_Enum.FAILED.value)
        self.assertEqual(failure_info["failed_stage"], stage)
        self.assertEqual(failure_info["error_message"], error_message)
        self.assertEqual(failure_info["experiment_type"], self.config["experiment_type"])
        self.assertIn("run_ids", failure_info)
