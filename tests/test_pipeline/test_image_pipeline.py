import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum

from pipeline.image_pipeline import ImagePipeline

class TestImagePipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock configuration
        self.config = {
            "dataset_path": "test_images/",
            "dataset_name": "test_dataset_img",
            "task_type": "image_classification",
            "logs": True,
            "no_classes": 2,
            "device": "cpu",
            "experiment_type": "image_experiment",
            "combinations": {
                "RESNET16": [
                    ("run_id_1", "PlattScaling", {"max_iter": [100, 200]}),
                    ("run_id_2", "IsotonicRegression", {})
                ]
            }
        }
            
        # Create pipeline instance with mocked ConfigurationManager
        with patch('pipeline.image_pipeline.ConfigurationManager') as mock_config:
            mock_config.return_value.random_seed = 42
            mock_config.return_value.n_bins = 10
            mock_config.return_value.conf_thresholds_list = [0.5, 0.7, 0.9]
            mock_config.return_value.epochs = 50  # Updated to match actual config
            mock_config.return_value.config_img = "mock_path"
            self.pipeline = ImagePipeline(self.config)

    def _create_mock_dataloader(self, num_samples=10):
        """Helper method to create mock DataLoader"""
        mock_dataset = [(torch.randn(3, 224, 224), torch.tensor(np.random.randint(0, 2))) 
                       for _ in range(num_samples)]
        return DataLoader(mock_dataset, batch_size=2)

    def test_initialization(self):
        """Test pipeline initialization"""
        self.assertIsNotNone(self.pipeline)
        self.assertEqual(self.pipeline.config, self.config)
        self.assertIsNotNone(self.pipeline.logger)
        self.assertEqual(len(self.pipeline.classifiers), 0)
        self.assertEqual(len(self.pipeline.results), 0)
        self.assertEqual(len(self.pipeline.failed_models), 0)

    @patch('pipeline.image_pipeline.ImagesSplitter')
    @patch('pipeline.image_pipeline.ImagePreprocessor')
    def test_load_preprocess_data(self, mock_preprocessor, mock_splitter):
        """Test data loading and preprocessing"""
        # Mock splitter
        mock_splitter_instance = Mock()
        train_imgs = ["train1.jpg", "train2.jpg", "train3.jpg"]
        val_imgs = ["val1.jpg", "val2.jpg"]
        test_imgs = ["test1.jpg", "test2.jpg"]
        mock_splitter_instance.split_dataset.return_value = (train_imgs, val_imgs, test_imgs)
        mock_splitter.return_value = mock_splitter_instance

        # Mock preprocessor
        mock_preprocessor_instance = Mock()
        mock_preprocessor_instance.fit_transform.return_value = self._create_mock_dataloader(3)
        mock_preprocessor_instance.transform.side_effect = [
            self._create_mock_dataloader(2),
            self._create_mock_dataloader(2)
        ]
        mock_preprocessor_instance.get_timing.return_value = {"preprocess_time": 0.2}
        mock_preprocessor.return_value = mock_preprocessor_instance

        # Test data loading
        self.pipeline.load_preprocess_data()

        # Assertions
        self.assertIsNotNone(self.pipeline.train_loader)
        self.assertIsNotNone(self.pipeline.val_loader)
        self.assertIsNotNone(self.pipeline.test_loader)
        self.assertEqual(self.pipeline.n_instances, 2)
        self.assertIsInstance(self.pipeline.split_ratios, dict)
        self.assertIsNotNone(self.pipeline.preprocessing_timing_info)

    @patch('pipeline.image_pipeline.ModelCache')
    def test_initialize_model(self, mock_model_cache):
        """Test model initialization"""
        # Mock model cache
        mock_model = Mock()
        mock_model_cache.get_model.return_value = mock_model

        self.pipeline.initialize_model()

        # Assertions
        self.assertEqual(len(self.pipeline.classifiers), 1)
        mock_model_cache.get_model.assert_called_once()
        self.assertNotIn("RESNET16", self.pipeline.failed_models)

    def test_train_model(self):
        """Test model training"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.training_metrics = {"accuracy": 0.9}
        mock_classifier.training_time = 0.5
        mock_classifier.train = Mock()
        
        # Mock ConfigurationManager
        with patch('pipeline.image_pipeline.config_manager') as mock_config_manager:
            mock_config_manager.epochs = 50  # Match the actual config value
            
            self.pipeline.classifiers = {"RESNET16": mock_classifier}
            self.pipeline.train_loader = self._create_mock_dataloader()
            self.pipeline.val_loader = self._create_mock_dataloader()

            self.pipeline.train_model()

            # Assertions
            mock_classifier.train.assert_called_once_with(
                dataloader=self.pipeline.train_loader,
                val_loader=self.pipeline.val_loader,
                epochs=50  # Updated to match the actual config value
            )
            self.assertIn("RESNET16", self.pipeline.results)
            self.assertEqual(
                self.pipeline.results["RESNET16"]["training_metrics"],
                {"accuracy": 0.9}
            )
            self.assertNotIn("RESNET16", self.pipeline.failed_models)

    def test_get_labels_from_loader(self):
        """Test label extraction from DataLoader"""
        mock_loader = self._create_mock_dataloader(5)
        labels = self.pipeline.get_labels_from_loader(mock_loader)
        
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(labels), 5)

    def test_evaluate_model(self):
        """Test model evaluation"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.predict.return_value = np.random.randint(0, 2, 10)
        mock_classifier.predict_prob.return_value = np.random.rand(10, 2)
        mock_classifier.testing_metrics = {}
        mock_classifier.testing_time_predictprob = 0.1

        self.pipeline.classifiers = {"RESNET16": mock_classifier}
        self.pipeline.test_loader = self._create_mock_dataloader(10)
        self.pipeline.results = {"RESNET16": {}}

        self.pipeline.evaluate_model()

        # Assertions
        mock_classifier.predict.assert_called_once()
        mock_classifier.predict_prob.assert_called_once()
        self.assertIn("testing_metrics", self.pipeline.results["RESNET16"])
        self.assertIn("predict_proba_time", self.pipeline.results["RESNET16"])
        self.assertIn("probabilities", self.pipeline.results["RESNET16"])
        self.assertNotIn("RESNET16", self.pipeline.failed_models)

    @patch('pipeline.image_pipeline.tune_all_calibration')
    def test_calibrate_model(self, mock_tune_calibration):
        """Test model calibration"""
        # Setup mock classifier
        mock_classifier = Mock()
        mock_classifier.predict_prob.return_value = np.random.rand(10, 2)
        
        # Setup pipeline attributes
        self.pipeline.classifiers = {"RESNET16": mock_classifier}
        self.pipeline.val_loader = self._create_mock_dataloader(10)
        self.pipeline.test_loader = self._create_mock_dataloader(10)
        self.pipeline.true_labels_test = np.random.randint(0, 2, 10)
        self.pipeline.n_instances = 10
        self.pipeline.split_ratios = {'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15}
        self.pipeline.preprocessing_timing_info = {"preprocess_time": 0.2}

        # Setup results
        self.pipeline.results = {
            "RESNET16": {
                "training_time": 0.1,
                "predict_proba_time": 0.1,
                "training_metrics": {"accuracy": 0.9},
                "testing_metrics": {"accuracy": 0.85},
                "probabilities": np.random.rand(10, 2)
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

        # Add experiment manager mock
        self.pipeline.experiment_manager = Mock()

        # Execute calibration
        self.pipeline.calibrate_model()

        # Assertions
        mock_classifier.predict_prob.assert_called_once()
        mock_tune_calibration.assert_called_once()
        self.pipeline.experiment_manager._save_results.assert_called()
        self.assertNotIn("RESNET16", self.pipeline.failed_models)

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
        self.assertEqual(failure_info["run_ids"], ["run_id_1", "run_id_2"])

if __name__ == '__main__':
    unittest.main()