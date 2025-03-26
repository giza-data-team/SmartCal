import unittest
import pandas as pd
import numpy as np
import re
import json
import os
from unittest.mock import Mock, patch, MagicMock
from pipeline.language_pipeline import LanguagePipeline
from config.enums.language_models_enum import ModelType
from config.enums.language_models_enum import LanguageModelsEnum
from sklearn.metrics import precision_score, recall_score, f1_score

class TestLanguagePipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock for experiment type
        mock_experiment_type = Mock()
        mock_experiment_type.value = "language_experiment"
        
        # Mock configuration
        self.config = {
            "dataset_path": "test_data.csv",
            "dataset_name": "test_dataset_lang",
            "task_type": "text_classification",
            "model_type": ModelType.WordEmbeddingModel,
            "logs": True,
            "experiment_type": mock_experiment_type,  # Use the mock here
            "combinations": {
                "BERT": [
                    (1, {"PlattScaling": {"max_iter": [100, 200]}}),
                    (2, {"IsotonicRegression": {}})
                ]
            }
        }
 
        # Create pipeline instance with mocked ConfigurationManager
        with patch('pipeline.language_pipeline.ConfigurationManager') as mock_config:
            mock_config.return_value.random_seed = 42
            mock_config.return_value.n_bins = 10
            mock_config.return_value.conf_thresholds_list = [0.5, 0.7, 0.9]
            mock_config.return_value.config_language = "mock_path"
            mock_config.return_value.split_ratios = [0.7, 0.15, 0.15]
            self.pipeline = LanguagePipeline(self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'text': ['sample text 1', 'sample text 2', 'sample text 3'],
            'label': [0, 1, 0]
        })

    def tearDown(self):
        """Clean up after each test"""
        # Remove any JSON files created during tests
        for file in os.listdir():
            if file.endswith('.json'):
                os.remove(file)

    def test_initialization(self):
        """Test pipeline initialization and status tracking"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.config, self.config)
        self.assertIsNotNone(self.pipeline.logger)
        self.assertEqual(len(self.pipeline.classifiers), 0)
        self.assertEqual(len(self.pipeline.results), 0)
        
        # Test pipeline status initialization
        self.assertEqual(self.pipeline.pipeline_status['status'], 'initialized')
        self.assertIsNone(self.pipeline.pipeline_status['failed_stage'])
        self.assertIsNone(self.pipeline.pipeline_status['error_message'])
        self.assertEqual(self.pipeline.pipeline_status['completed_stages'], [])

    @patch('pipeline.language_pipeline.LanguageSplitter')
    @patch('pipeline.language_pipeline.LanguagePreprocessor')
    def test_load_preprocess_data(self, mock_preprocessor, mock_splitter):
        """Test data loading and preprocessing"""
        # Mock splitter
        mock_splitter_instance = Mock()
        train_data = self.sample_data.iloc[:2]
        val_data = self.sample_data.iloc[2:3]
        test_data = self.sample_data.iloc[2:3]
        mock_splitter_instance.split_dataset.return_value = (train_data, val_data, test_data)
        mock_splitter_instance.get_timing.return_value = {"split_time": 0.1}
        mock_splitter.return_value = mock_splitter_instance

        # Mock preprocessor
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.fit_transform.return_value = (
            ['processed text 1', 'processed text 2'],
            ['__label__0', '__label__1']
        )
        mock_preprocessor_instance.transform.side_effect = [
            (['processed text 3'], ['__label__0']),
            (['processed text 3'], ['__label__0'])
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
        self.assertEqual(self.pipeline.n_instances, 1)

    def test_failed_data_loading(self):
        """Test error handling during data loading"""
        # Mock read_csv to raise an error
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            success = self.pipeline.run()
            
            # Assert pipeline failed
            self.assertFalse(success)
            self.assertEqual(self.pipeline.pipeline_status['status'], 'failed')
            self.assertEqual(self.pipeline.pipeline_status['failed_stage'], 'load_preprocess_data')
            self.assertIn("File not found", self.pipeline.pipeline_status['error_message'])
            
            # Check if failure was logged to file
            failure_files = [f for f in os.listdir() if f.startswith('pipeline_failure')]
            self.assertEqual(len(failure_files), 1)
            
            # Verify failure file contents
            with open(failure_files[0], 'r') as f:
                failure_data = json.load(f)
                self.assertEqual(failure_data['status'], 'failed')
                self.assertEqual(failure_data['failed_stage'], 'load_preprocess_data')
                self.assertIn("File not found", failure_data['error_message'])

    @patch('pipeline.language_pipeline.ModelCache')
    def test_initialize_model(self, mock_model_cache):
        """Test model initialization"""
        # Mock model cache
        mock_model = Mock()
        mock_model_cache.get_model.return_value = mock_model

        self.pipeline.initialize_model()

        # Assertions
        self.assertEqual(len(self.pipeline.classifiers), 1)
        mock_model_cache.get_model.assert_called_once_with(
            model_enum="BERT",
            task_type=self.config["task_type"],
            seed=self.pipeline.random_seed
        )

    def test_failed_model_initialization(self):
        """Test error handling during model initialization"""
        self.pipeline.load_preprocess_data = Mock()
        self.pipeline.initialize_model = Mock(side_effect=ValueError("Invalid model configuration"))
        
        success = self.pipeline.run()
        
        # Assert pipeline failed at model initialization
        self.assertFalse(success)
        self.assertEqual(self.pipeline.pipeline_status['status'], 'failed')
        self.assertEqual(self.pipeline.pipeline_status['failed_stage'], 'initialize_model')
        self.assertIn("Invalid model configuration", self.pipeline.pipeline_status['error_message'])
        self.assertEqual(len(self.pipeline.pipeline_status['completed_stages']), 1)

    def test_train_model(self):
        """Test model training"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.training_metrics = {"accuracy": 0.9}
        mock_classifier.training_time = 0.5
        
        self.pipeline.classifiers = {"BERT": mock_classifier}
        self.pipeline.X_train = ['processed text 1', 'processed text 2']
        self.pipeline.y_train = ['__label__0', '__label__1']
        self.pipeline.X_valid = ['processed text 1', 'processed text 2']
        self.pipeline.y_valid = ['__label__0', '__label__1']

        self.pipeline.train_model()

        # Assertions
        mock_classifier.train.assert_called_once_with(
            self.pipeline.X_train,
            self.pipeline.y_train,
            self.pipeline.X_valid,
            self.pipeline.y_valid,
        )
        self.assertIn("BERT", self.pipeline.results)
        self.assertEqual(
            self.pipeline.results["BERT"]["training_metrics"],
            {"accuracy": 0.9}
        )

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = np.array([0, 1])
        mock_classifier.predict_prob.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_classifier.testing_metrics = {}
        mock_classifier.testing_time_predictprob = 0.1

        # Setup pipeline attributes
        self.pipeline.classifiers = {"BERT": mock_classifier}
        self.pipeline.X_test = ['processed text 1', 'processed text 2']
        self.pipeline.y_test = ['__label__0', '__label__1']
        self.pipeline.results = {"BERT": {}}
        self.pipeline.config["model_type"] = ModelType.WordEmbeddingModel
        
        # Set up ground truth labels
        self.pipeline.ground_truth_val = [0, 1]  # Add this
        self.pipeline.ground_truth_test = [0, 1]  # Add this
        self.pipeline.true_labels_val = [0, 1]
        self.pipeline.true_labels_tst = [0, 1]

        self.pipeline.evaluate_model()

        # Assertions
        mock_classifier.predict.assert_called_once()
        mock_classifier.predict_prob.assert_called_once()
        self.assertIn("testing_metrics", self.pipeline.results["BERT"])
        self.assertIn("predict_proba_time", self.pipeline.results["BERT"])

    @patch('pipeline.language_pipeline.tune_all_calibration')
    @patch('builtins.open')
    def test_calibrate_model(self, mock_open, mock_tune_calibration):
        """Test model calibration"""
        # Setup mock classifier
        mock_classifier = Mock()
        mock_classifier.predict_prob.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        # Use LanguageModelsEnum.TINYBERT
        self.pipeline.classifiers = {LanguageModelsEnum.TINYBERT: mock_classifier}
        self.pipeline.X_valid = ['processed text 1', 'processed text 2']
        self.pipeline.true_labels_val = [0, 1]
        self.pipeline.true_labels_tst = [0, 1]
        self.pipeline.n_instances = 2
        self.pipeline.preprocessing_timing_info = {"preprocess_time": 0.2}
        
        # Setup experiment type
        mock_experiment_type = Mock()
        mock_experiment_type.value = "language_experiment"
        self.pipeline.config["experiment_type"] = mock_experiment_type
        
        # Setup results dictionary
        self.pipeline.results = {
            LanguageModelsEnum.TINYBERT: {
                "training_time": 0.1,
                "predict_proba_time": 0.1,
                "training_metrics": {"accuracy": 0.9},
                "testing_metrics": {
                    'accuracy': 0.85,
                    'loss': 0.3,
                    'ece': 0.1,
                    'mce': 0.2,
                    'conf_ece': (0.15, 0.18, 0.20)
                },
                "probabilities": np.array([[0.8, 0.2], [0.3, 0.7]])
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
                    "ece": 0.08,
                    "precision_micro": 0.88,
                    "precision_macro": 0.87,
                    "recall_micro": 0.86,
                    "recall_macro": 0.85,
                    "f1_micro": 0.87,
                    "f1_macro": 0.86
                },
                "calibrated_metrics_tst_set": {
                    "accuracy": 0.86,
                    "loss": 0.27,
                    "ece": 0.09
                }
            }
        }

        # Mock the file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        self.pipeline.calibrate_model()

        # Assertions
        mock_classifier.predict_prob.assert_called_once()
        mock_tune_calibration.assert_called_once()
        
        # Verify final results structure
        self.assertIsNotNone(self.pipeline.final_results)
        
        # The final_results should have a models_results key
        self.assertIn('models_results', self.pipeline.final_results)
        
        # Get the models_results dictionary
        models_results = self.pipeline.final_results['models_results']
        
        # Verify TINYBERT exists in the results
        self.assertIn('TINYBERT', models_results)
        
        # Get TINYBERT results
        tinybert_results = models_results['TINYBERT']
        
        # Verify the structure of TINYBERT results
        self.assertIn('calibration_results', tinybert_results)
        self.assertIn('PlattScaling', tinybert_results['calibration_results'])
        
        # Verify metrics exist
        platt_results = tinybert_results['calibration_results']['PlattScaling']
        self.assertIn('calibrated_metrics_val_set', platt_results)
        
        # Verify specific metrics
        metrics = platt_results['calibrated_metrics_val_set']
        self.assertIn('precision_micro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_micro', metrics)
        
        # Verify values of some metrics
        self.assertEqual(metrics['precision_micro'], 0.88)
        self.assertEqual(metrics['recall_macro'], 0.85)
        self.assertEqual(metrics['f1_micro'], 0.87)
        
    def test_successful_pipeline_execution(self):
        """Test successful pipeline execution and status tracking"""
        # Mock all pipeline methods to succeed
        self.pipeline.load_preprocess_data = Mock()
        self.pipeline.initialize_model = Mock()
        self.pipeline.train_model = Mock()
        self.pipeline.evaluate_model = Mock()
        self.pipeline.calibrate_model = Mock()

        success = self.pipeline.run()

        # Assert pipeline succeeded
        self.assertTrue(success)
        self.assertEqual(self.pipeline.pipeline_status['status'], 'completed')
        self.assertEqual(len(self.pipeline.pipeline_status['completed_stages']), 5)
        self.assertIn('load_preprocess_data', self.pipeline.pipeline_status['completed_stages'])
        self.assertIn('calibrate_model', self.pipeline.pipeline_status['completed_stages'])

    def test_failed_data_loading(self):
        """Test error handling during data loading"""
        with patch('pandas.read_csv', side_effect=FileNotFoundError("File not found")):
            result = self.pipeline.run()
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['failed_stage'], 'load_preprocess_data')

    def test_failed_model_initialization(self):
        """Test error handling during model initialization"""
        self.pipeline.load_preprocess_data = Mock()
        self.pipeline.initialize_model = Mock(side_effect=ValueError("Invalid model configuration"))
        
        result = self.pipeline.run()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failed_stage'], 'initialize_model')

    def test_execute_stage(self):
        """Test stage execution with error handling"""
        # Test successful stage execution
        def successful_stage():
            return True
        
        success, failure_info = self.pipeline.execute_stage("test_stage", successful_stage)
        self.assertTrue(success)
        self.assertIsNone(failure_info)
        
        # Test failed stage execution
        def failing_stage():
            raise ValueError("Test error")
        
        success, failure_info = self.pipeline.execute_stage("failing_stage", failing_stage)
        self.assertFalse(success)
        self.assertEqual(failure_info['failed_stage'], "failing_stage")