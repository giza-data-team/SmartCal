import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

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
            "experiment_type": Mock(value="test_experiment"),  # Add this line
            "combinations": {
                "RANDOM_FOREST": [
                    (1, {"PlattScaling": {"max_iter": [100, 200]}}),
                    (2, {"IsotonicRegression": {}})
                ]
            }
        }
        
        # Create pipeline instance with mocked ConfigurationManager
        with patch('pipeline.tabular_pipeline.ConfigurationManager') as mock_config:
            mock_config.return_value.random_seed = 42
            mock_config.return_value.n_bins = 10
            mock_config.return_value.conf_thresholds_list = [0.5, 0.7, 0.9]
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
        mock_splitter_instance.get_timing.return_value = {"split_time": 0.1}
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

    def test_train_model(self):
        """Test model training"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.training_metrics = {"accuracy": 0.9}
        mock_classifier.training_time = 0.5
        
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

        # Updated assertions to match actual implementation
        self.assertIn("testing_metrics", self.pipeline.results["RANDOM_FOREST"])
        self.assertIn("predict_proba_time", self.pipeline.results["RANDOM_FOREST"])
        self.assertIn("probabilities", self.pipeline.results["RANDOM_FOREST"])

    @patch('pipeline.tabular_pipeline.tune_all_calibration')
    def test_calibrate_model(self, mock_tune_calibration):
        """Test model calibration"""
        # Setup mock classifier
        mock_classifier = Mock()
        mock_classifier.predict_prob.return_value = np.random.rand(20, 2)
        
        # Create mock predictions and probabilities
        mock_predictions = np.random.randint(0, 2, 20)
        mock_probabilities = np.random.rand(20, 2)
        
        # Setup mock testing metrics
        mock_testing_metrics = {
            'accuracy': 0.85,
            'loss': 0.3,
            'ece': 0.1,
            'mce': 0.2,
            'conf_ece': (0.15, 0.18, 0.20)
        }
        
        self.pipeline.classifiers = {"RANDOM_FOREST": mock_classifier}
        self.pipeline.X_valid = np.random.rand(20, 2)
        self.pipeline.y_valid = np.random.randint(0, 2, 20)
        self.pipeline.X_test = np.random.rand(20, 2)
        self.pipeline.y_test = np.random.randint(0, 2, 20)
        
        # Set n_instances attribute
        self.pipeline.n_instances = 20
        
        # Set split_timing_info and preprocessing_timing_info
        self.pipeline.split_timing_info = {"split_time": 0.1}
        self.pipeline.preprocessing_timing_info = {"preprocess_time": 0.2}
        
        # Setup complete results dictionary with all required keys
        self.pipeline.results = {
        "RANDOM_FOREST": {
            "training_time": 0.1,
            "predict_proba_time": 0.1,  
            "training_metrics": {"accuracy": 0.9},
            "testing_metrics": mock_testing_metrics,
            "probabilities": mock_probabilities
        }
    }

        # Mock calibration results
        mock_tune_calibration.return_value = {
            "PlattScaling": {
                "best_hyperparams": {"max_iter": 100},
                "best_ece": 0.05,
                "calibrated_metrics_val_set": {
                    "accuracy": 0.87,
                    "loss": 0.25,
                    "ece": 0.08
                },
                "calibrated_metrics_tst_set": {
                    "accuracy": 0.86,
                    "loss": 0.27,
                    "ece": 0.09
                }
            }
        }

        # Mock ConfigurationManager's split_ratios
        with patch('pipeline.tabular_pipeline.config_manager') as mock_config_manager:
            mock_config_manager.split_ratios = [0.7, 0.15, 0.15]
            
            self.pipeline.calibrate_model()

        # Assertions
        mock_classifier.predict_prob.assert_called()
        mock_tune_calibration.assert_called_once_with(
            mock_classifier.predict_prob.return_value,  # uncalibrated_valid_probs
            mock_probabilities,  # uncalibrated_test_probs
            self.pipeline.y_valid,
            self.pipeline.y_test,
            self.pipeline,
            "RANDOM_FOREST",
            self.pipeline.config,
            None
        )

        # Additional assertions to verify the results
        self.assertIn("RANDOM_FOREST", self.pipeline.results)
        results = self.pipeline.results["RANDOM_FOREST"]
        self.assertIn("training_metrics", results)
        self.assertIn("testing_metrics", results)
        self.assertIn("probabilities", results)