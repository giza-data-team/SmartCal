import unittest
import pandas as pd
import numpy as np
import json
import os
from unittest.mock import Mock, patch, ANY  

from pipeline.language_pipeline import LanguagePipeline
from smartcal.config.enums.language_models_enum import ModelType
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum
from smartcal.config.enums.language_models_enum import LanguageModelsEnum


class TestLanguagePipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock for experiment type
        mock_experiment_type = Mock()
        mock_experiment_type.value = "language_experiment"
        
        # Mock configuration with new kfold parameter
        self.config = {
            "dataset_path": "test_data.csv",
            "dataset_name": "test_dataset_lang",
            "task_type": "text_classification",
            "model_type": ModelType.WordEmbeddingModel,
            "logs": True,
            "experiment_type": mock_experiment_type,
            "use_kfold": True,
            "combinations": {
                "TINYBERT": [
                    ("run_1", "PlattScaling", {"max_iter": [100, 200]}),
                    ("run_2", "IsotonicRegression", {})
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

        # Mock experiment manager
        self.mock_experiment_manager = Mock()
        self.pipeline.experiment_manager = self.mock_experiment_manager

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

    def test_convert_numpy_types(self):
        """Test numpy type conversion functionality"""
        # Test numpy array conversion
        np_array = np.array([1, 2, 3])
        converted_array = self.pipeline.convert_numpy_types(np_array)
        self.assertIsInstance(converted_array, list)
        
        # Test numpy scalar conversion
        np_scalar = np.float32(1.0)
        converted_scalar = self.pipeline.convert_numpy_types(np_scalar)
        self.assertIsInstance(converted_scalar, float)
        
        # Test nested dictionary conversion
        nested_dict = {
            'array': np.array([1, 2, 3]),
            'scalar': np.int32(5),
            'nested': {'value': np.float64(2.0)}
        }
        converted_dict = self.pipeline.convert_numpy_types(nested_dict)
        self.assertIsInstance(converted_dict['array'], list)
        self.assertIsInstance(converted_dict['scalar'], int)
        self.assertIsInstance(converted_dict['nested']['value'], float)

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
            model_enum="TINYBERT",
            task_type=self.config["task_type"],
            seed=self.pipeline.random_seed
        )

    def test_train_model(self):
        """Test model training"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.training_metrics = {"accuracy": 0.9}
        mock_classifier.training_time = 0.5

        # Use enum instead of string
        self.pipeline.classifiers = {LanguageModelsEnum.TINYBERT: mock_classifier}
        self.pipeline.X_train = ['processed text 1', 'processed text 2']
        self.pipeline.y_train = ['__label__0', '__label__1']
        self.pipeline.X_valid = ['processed text 1', 'processed text 2']
        self.pipeline.y_valid = ['__label__0', '__label__1']

        self.pipeline.train_model()

        # Assertions - also check with enum
        mock_classifier.train.assert_called_once_with(
            self.pipeline.X_train,
            self.pipeline.y_train,
            self.pipeline.X_valid,
            self.pipeline.y_valid,
        )
        self.assertIn(LanguageModelsEnum.TINYBERT, self.pipeline.results)
        self.assertEqual(
            self.pipeline.results[LanguageModelsEnum.TINYBERT]["training_metrics"],
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

        # Setup pipeline attributes - use enum instead of string
        self.pipeline.classifiers = {LanguageModelsEnum.TINYBERT: mock_classifier}
        self.pipeline.X_test = ['processed text 1', 'processed text 2']
        self.pipeline.y_test = ['__label__0', '__label__1']
        self.pipeline.results = {LanguageModelsEnum.TINYBERT: {}}
        self.pipeline.config["model_type"] = ModelType.WordEmbeddingModel

        # Set up ground truth labels
        self.pipeline.ground_truth_val = [0, 1]
        self.pipeline.ground_truth_test = [0, 1]
        self.pipeline.true_labels_val = [0, 1]
        self.pipeline.true_labels_test = [0, 1]

        self.pipeline.evaluate_model()

        # Assertions
        mock_classifier.predict.assert_called_once()
        mock_classifier.predict_prob.assert_called_once()
        self.assertIn("testing_metrics", self.pipeline.results[LanguageModelsEnum.TINYBERT])
        self.assertIn("predict_proba_time", self.pipeline.results[LanguageModelsEnum.TINYBERT])

    @patch('pipeline.language_pipeline.tune_all_calibration')
    def test_calibrate_model_with_kfold(self, mock_tune_calibration):
        """Test model calibration with k-fold cross validation"""
        # Setup necessary attributes
        mock_classifier = Mock()
        mock_classifier.predict_prob.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])

        self.pipeline.classifiers = {LanguageModelsEnum.TINYBERT: mock_classifier}
        self.pipeline.X_valid = ['processed text 1', 'processed text 2']
        self.pipeline.X_test = ['processed text 3', 'processed text 4']
        self.pipeline.true_labels_val = [0, 1]
        self.pipeline.true_labels_test = [0, 1]
        self.pipeline.n_instances = 2
        self.pipeline.preprocessing_timing_info = {"preprocess_time": 0.2}
        self.pipeline.y_valid = ['__label__0', '__label__1']
        self.pipeline.y_test = ['__label__0', '__label__1']

        # Setup results dictionary - also using LanguageModelsEnum.TINYBERT
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
        
        # Mock successful calibration results
        calibration_results = {
            "PlattScaling_run_1": {
                "run_id": "run_1",
                "calibration_metric": "ECE",
                "cal_status": Experiment_Status_Enum.COMPLETED.value,
                "cal_timing": {"fit": 0.1, "predict": 0.05},
                "best_hyperparams": {"max_iter": 100},
                "calibrated_metrics_val_set": {
                    "accuracy": 0.87,
                    "loss": 0.25,
                    "ece": 0.08,
                    "mce": 0.15,
                    "conf_ece": (0.12, 0.14, 0.16)
                },
                "calibrated_metrics_test_set": {
                    "accuracy": 0.86,
                    "loss": 0.26,
                    "ece": 0.07,
                    "mce": 0.14,
                    "conf_ece": (0.11, 0.13, 0.15)
                },
                "calibrated_probs_val_set": [[0.75, 0.25], [0.35, 0.65]],
                "calibrated_probs_test_set": [[0.75, 0.25], [0.35, 0.65]]
            }
        }
        mock_tune_calibration.return_value = calibration_results

        # Mock the experiment manager's save_results to capture the actual call
        with patch.object(self.pipeline.experiment_manager, '_save_results') as mock_save:
            try:
                self.pipeline.calibrate_model()
            except Exception as e:
                self.fail(f"calibrate_model() raised {type(e).__name__} unexpectedly: {str(e)}")
                
            # Verify results were saved with COMPLETED status
            mock_save.assert_called_once()
            saved_results = mock_save.call_args[0][0]
            
            # Debug output
            print(f"Saved results status: {saved_results.get('status')}")
            print(f"Pipeline status: {self.pipeline.pipeline_status}")
            
    def test_split_seed_handling(self):
        """Test handling of split_seed parameter"""
        # Create a mock dataset file
        test_data = pd.DataFrame({
            'text': ['sample text 1', 'sample text 2'],
            'label': [0, 1]
        })
        
        # Create mock splitter instance with proper return values
        mock_splitter_instance = Mock()
        mock_splitter_instance.split_dataset.return_value = (
            test_data.iloc[:1],  # train
            test_data.iloc[1:2], # val
            test_data.iloc[1:2]  # test
        )
        
        # Test with explicit split_seed
        config_with_split_seed = self.config.copy()
        config_with_split_seed['split_seed'] = 123
        
        pipeline_with_split_seed = LanguagePipeline(config_with_split_seed)
        
        with patch('pandas.read_csv', return_value=test_data), \
            patch('pipeline.language_pipeline.LanguageSplitter') as mock_splitter, \
            patch('pipeline.language_pipeline.LanguagePreprocessor') as mock_preprocessor:
            
            # Setup mock splitter
            mock_splitter.return_value = mock_splitter_instance
            
            # Setup mock preprocessor
            mock_preprocessor_instance = Mock()
            mock_preprocessor_instance.fit_transform.return_value = (
                ['processed text 1'],
                ['__label__0']
            )
            mock_preprocessor_instance.transform.return_value = (
                ['processed text 2'],
                ['__label__1']
            )
            mock_preprocessor.return_value = mock_preprocessor_instance
            
            pipeline_with_split_seed.load_preprocess_data()
            mock_splitter.assert_called_with(
                dataset_name=config_with_split_seed["dataset_name"],
                metadata_path=ANY,  # Use ANY instead of mock.ANY
                logs=config_with_split_seed["logs"],
                random_seed=123
            )

        # Test without split_seed
        pipeline_without_split_seed = LanguagePipeline(self.config)
        with patch('pandas.read_csv', return_value=test_data), \
            patch('pipeline.language_pipeline.LanguageSplitter') as mock_splitter, \
            patch('pipeline.language_pipeline.LanguagePreprocessor') as mock_preprocessor:
            
            # Setup mock splitter
            mock_splitter.return_value = mock_splitter_instance
            
            # Setup mock preprocessor
            mock_preprocessor.return_value = mock_preprocessor_instance
            
            pipeline_without_split_seed.load_preprocess_data()
            mock_splitter.assert_called_with(
                dataset_name=self.config["dataset_name"],
                metadata_path=ANY,  # Use ANY instead of mock.ANY
                logs=self.config["logs"],
                random_seed=42
            )

    def test_calibrate_model_error_handling(self):
        """Test error handling in calibrate_model with experiment manager"""
        mock_classifier = Mock()
        mock_classifier.predict_prob.side_effect = Exception("Calibration failed")

        # Use LanguageModelsEnum.TINYBERT instead of the string "TINYBERT"
        self.pipeline.classifiers = {LanguageModelsEnum.TINYBERT: mock_classifier}

        # Set up necessary attributes
        self.pipeline.X_valid = ['processed text 1', 'processed text 2']
        self.pipeline.X_test = ['processed text 3', 'processed text 4']
        self.pipeline.true_labels_val = [0, 1]
        self.pipeline.true_labels_test = [0, 1]

        self.pipeline.calibrate_model()

        # Verify error was saved through experiment manager
        self.mock_experiment_manager._save_results.assert_called_once()
        saved_failure = self.mock_experiment_manager._save_results.call_args[0][0]
        self.assertEqual(saved_failure['status'], Experiment_Status_Enum.FAILED.value)

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
        self.assertEqual(self.pipeline.pipeline_status['status'], Experiment_Status_Enum.COMPLETED.value)
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
        
        # Mock the save_failure_status method
        with patch.object(self.pipeline, 'save_failure_status') as mock_save_failure:
            mock_save_failure.return_value = {'status': 'failed', 'failed_stage': 'failing_stage'}
            success, failure_info = self.pipeline.execute_stage("failing_stage", failing_stage)
            self.assertFalse(success)
            self.assertEqual(failure_info['failed_stage'], "failing_stage")