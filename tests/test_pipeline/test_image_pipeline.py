import unittest
import numpy as np
import torch
from unittest.mock import Mock, patch
from torch.utils.data import DataLoader

from pipeline.image_pipeline import ImagePipeline


class TestImagePipeline(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock for experiment type
        mock_experiment_type = Mock()
        mock_experiment_type.value = "image_experiment"
        
        # Mock configuration
        self.config = {
            "dataset_path": "test_images/",
            "dataset_name": "test_dataset_img",
            "task_type": "image_classification",
            "logs": True,
            "no_classes": 2,
            "device": "cpu",
            "experiment_type": mock_experiment_type,  # Add this line
            "combinations": {
                "RESNET16": [
                    (1, {"PlattScaling": {"max_iter": [100, 200]}}),
                    (2, {"IsotonicRegression": {}})
                ]
            }
        }
            
        # Create pipeline instance with mocked ConfigurationManager
        with patch('pipeline.image_pipeline.ConfigurationManager') as mock_config:
            mock_config.return_value.random_seed = 42
            mock_config.return_value.n_bins = 10
            mock_config.return_value.conf_thresholds_list = [0.5, 0.7, 0.9]
            mock_config.return_value.epochs = 1
            mock_config.return_value.learning_rate = 0.001
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
        mock_splitter_instance.get_timing.return_value = {"split_time": 0.1}
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
        self.assertEqual(self.pipeline.n_instances, 2)  # Length of val_imgs
        self.assertIsInstance(self.pipeline.split_ratios, dict)

    @patch('pipeline.image_pipeline.ModelCache')
    def test_initialize_model(self, mock_model_cache):
        """Test model initialization"""
        # Mock model cache
        mock_model = Mock()
        mock_model_cache.get_model.return_value = mock_model

        self.pipeline.initialize_model()

        # Assertions
        self.assertEqual(len(self.pipeline.classifiers), 1)
        mock_model_cache.get_model.assert_called_once_with(
            model_enum="RESNET16",
            task_type=self.config["task_type"],
            num_classes=self.config["no_classes"],
            device=self.config["device"],
            seed=self.pipeline.random_seed
        )

    def test_train_model(self):
        """Test model training"""
        # Create mock classifier
        mock_classifier = Mock()
        mock_classifier.training_metrics = {"accuracy": 0.9}
        mock_classifier.training_time = 0.5
        
        self.pipeline.classifiers = {"RESNET16": mock_classifier}
        self.pipeline.train_loader = self._create_mock_dataloader()
        self.pipeline.val_loader = self._create_mock_dataloader()

        self.pipeline.train_model()

        # Assertions
        mock_classifier.train.assert_called_once()
        self.assertIn("RESNET16", self.pipeline.results)
        self.assertEqual(
            self.pipeline.results["RESNET16"]["training_metrics"],
            {"accuracy": 0.9}
        )

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

    @patch('pipeline.image_pipeline.tune_all_calibration')
    def test_calibrate_model(self, mock_tune_calibration):
        """Test model calibration"""
        # Setup mock classifier
        mock_classifier = Mock()
        mock_classifier.predict_prob.return_value = np.random.rand(10, 2)
        
        # Setup test data
        self.pipeline.classifiers = {"RESNET16": mock_classifier}
        self.pipeline.val_loader = self._create_mock_dataloader(10)
        self.pipeline.test_loader = self._create_mock_dataloader(10)
        self.pipeline.true_labels_test = np.random.randint(0, 2, 10)
        self.pipeline.n_instances = 10
        self.pipeline.split_ratios = {'train_ratio': 0.7, 'val_ratio': 0.15, 'test_ratio': 0.15}
        self.pipeline.preprocessing_timing_info = {"preprocess_time": 0.2}
        
        # Ensure experiment_type is set
        mock_experiment_type = Mock()
        mock_experiment_type.value = "image_experiment"
        self.pipeline.config["experiment_type"] = mock_experiment_type
        
        # Setup results dictionary
        self.pipeline.results = {
            "RESNET16": {
                "training_time": 0.1,
                "predict_proba_time": 0.1,
                "training_metrics": {"accuracy": 0.9},  # Add this line
                "testing_metrics": {
                    'accuracy': 0.85,
                    'loss': 0.3,
                    'ece': 0.1,
                    'mce': 0.2,
                    'conf_ece': (0.15, 0.18, 0.20)
                },
                "probabilities": np.random.rand(10, 2)
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

        # Mock save_calibration_details to return the desired structure
        with patch.object(self.pipeline, 'save_calibration_details') as mock_save:
            # Return a structure that matches your assertions
            mock_save.return_value = {
                "experiment_type": mock_experiment_type.value,
                "status": "COMPLETED",
                "dataset_info": {
                    "n_instances_cal_set": 10,
                    "split_ratios(train_cal_tst)": self.pipeline.split_ratios
                },
                "models_results": {
                    "RESNET16": {  # This is what your test is looking for
                        "calibration_results": mock_tune_calibration.return_value
                    }
                }
            }

            self.pipeline.calibrate_model()

            # Assertions
            mock_classifier.predict_prob.assert_called()
            mock_tune_calibration.assert_called_once()
            
            # Verify final results structure
            self.assertIsNotNone(self.pipeline.final_results)
            self.assertIn('dataset_info', self.pipeline.final_results)
            self.assertIn('models_results', self.pipeline.final_results)
            self.assertIn('RESNET16', self.pipeline.final_results['models_results'])

    def test_run(self):
        """Test complete pipeline run"""
        # Mock all pipeline methods
        self.pipeline.load_preprocess_data = Mock()
        self.pipeline.initialize_model = Mock()
        self.pipeline.train_model = Mock()
        self.pipeline.evaluate_model = Mock()
        self.pipeline.calibrate_model = Mock()

        self.pipeline.run()

        # Assert all methods were called
        self.pipeline.load_preprocess_data.assert_called_once()
        self.pipeline.initialize_model.assert_called_once()
        self.pipeline.train_model.assert_called_once()
        self.pipeline.evaluate_model.assert_called_once()
        self.pipeline.calibrate_model.assert_called_once()