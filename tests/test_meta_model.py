import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

from smartcal.meta_model.meta_model_factory import MetaModelFactory
from smartcal.meta_model.meta_model import MetaModel


class TestMetaModelFactory(unittest.TestCase):
    def setUp(self):
        self.factory = MetaModelFactory()

    @patch('smartcal.config.configuration_manager.configuration_manager.ConfigurationManager')
    def test_create_model_with_default_params(self, mock_config):
        # Setup mock config
        mock_config.return_value.metric = 'ECE'
        mock_config.return_value.k_recommendations = 3
        
        # Create model with default parameters
        model = self.factory.create_model()
        
        # Verify it's a meta-model
        self.assertIsInstance(model, MetaModel)

    @patch('smartcal.config.configuration_manager.configuration_manager.ConfigurationManager')
    def test_create_model_with_custom_params(self, mock_config):
        # Setup mock config
        mock_config.return_value.metric = 'ECE'
        mock_config.return_value.k_recommendations = 3
        
        # Create model with custom parameters
        model = self.factory.create_model(metric='brier_score', top_n=5)
        
        # Verify it's a meta-model
        self.assertIsInstance(model, MetaModel)


class TestMetaModel(unittest.TestCase):
    def setUp(self):
        self.factory = MetaModelFactory()
        self.test_input = {
            'num_classes': 5,
            'num_instances': 1000,
            'dataset_type': 0,
            'class_imbalance_ratio': 0.0,
            'actual_predictions_entropy': 0.0,
            'Confidence_Mean': 0.5,
            'Confidence_Median': 0.5,
            'Confidence_Std': 0.1,
            'Confidence_Var': 0.01,
            'Confidence_Entropy': 1.0,
            'Confidence_Skewness': 0.0,
            'Confidence_Kurtosis': 0.0,
            'Confidence_Min': 0.0,
            'Confidence_Max': 1.0,
            'Classification_Log_loss': 0.5,
            'Classification_Accuracy': 0.75,
            'Classification_Precision_Micro': 0.7,
            'Classification_Precision_Macro': 0.65,
            'Classification_Precision_Weighted': 0.68,
            'Classification_Recall_Micro': 0.71,
            'Classification_Recall_Macro': 0.64,
            'Classification_Recall_Weighted': 0.67,
            'Classification_F1_Micro': 0.70,
            'Classification_F1_Macro': 0.63,
            'Classification_F1_Weighted': 0.66,
            'ECE_before': 0.1,
            'MCE_before': 0.2,
            'ConfECE_before': 0.05,
            'brier_score_before': 0.15,
            'Wasserstein_Mean': 0.2,
            'Wasserstein_Median': 0.19,
            'Wasserstein_Std': 0.01,
            'Wasserstein_Var': 0.0001,
            'Wasserstein_Entropy': 1.0,
            'Wasserstein_Skewness': 0.0,
            'Wasserstein_Kurtosis': 0.0,
            'Wasserstein_Min': 0.0,
            'Wasserstein_Max': 1.0,
            'KL_Divergence_Mean': 0.1,
            'KL_Divergence_Median': 0.1,
            'KL_Divergence_Std': 0.01,
            'KL_Divergence_Var': 0.0001,
            'KL_Divergence_Entropy': 1.1,
            'KL_Divergence_Skewness': 0.0,
            'KL_Divergence_Kurtosis': 0.0,
            'KL_Divergence_Min': 0.0,
            'KL_Divergence_Max': 1.0,
            'Jensen_Shannon_Mean': 0.2,
            'Jensen_Shannon_Median': 0.2,
            'Jensen_Shannon_Std': 0.01,
            'Jensen_Shannon_Var': 0.0001,
            'Jensen_Shannon_Entropy': 1.0,
            'Jensen_Shannon_Skewness': 0.0,
            'Jensen_Shannon_Kurtosis': 0.0,
            'Jensen_Shannon_Min': 0.0,
            'Jensen_Shannon_Max': 1.0,
            'Bhattacharyya_Mean': 0.2,
            'Bhattacharyya_Median': 0.2,
            'Bhattacharyya_Std': 0.01,
            'Bhattacharyya_Var': 0.0001,
            'Bhattacharyya_Entropy': 1.0,
            'Bhattacharyya_Skewness': 0.0,
            'Bhattacharyya_Kurtosis': 0.0,
            'Bhattacharyya_Min': 0.0,
            'Bhattacharyya_Max': 1.0
        }

    def test_invalid_top_n(self):
        """Test that creating a model with invalid top_n values raises ValueError"""
        with self.assertRaises(ValueError):
            self.factory.create_model(top_n=0)
        with self.assertRaises(ValueError):
            self.factory.create_model(top_n=13)

    @patch('smartcal.meta_model.meta_model.MetaModel._load_component')
    def test_model_loading(self, mock_load):
        """Test model and label encoder loading behavior"""
        # Mock successful loading
        mock_model = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.classes_ = np.array(['model1', 'model2', 'model3'])
        mock_load.side_effect = [mock_model, mock_encoder]
        
        model = self.factory.create_model()
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.label_encoder)

        # Mock failed loading
        mock_load.side_effect = [None, None]
        model = self.factory.create_model()
        self.assertIsNone(model.model)
        self.assertIsNone(model.label_encoder)

    def test_prediction_with_different_metrics(self):
        """Test predictions with different calibration metrics"""
        metrics = ['ECE', 'MCE', 'brier_score']
        for metric in metrics:
            model = self.factory.create_model(metric=metric)
            predictions = model.predict_best_model(self.test_input)
            
            self.assertIsInstance(predictions, list)
            self.assertTrue(len(predictions) > 0)
            self.assertAlmostEqual(sum(float(pred[1]) for pred in predictions), 1.0, places=6)

    def test_input_validation(self):
        """Test prediction with invalid input features"""
        model = self.factory.create_model()
        
        # Test with missing required features
        invalid_input = self.test_input.copy()
        del invalid_input['num_classes']
        with self.assertRaises(Exception):
            model.predict_best_model(invalid_input)
        
        # Test with invalid feature type
        invalid_input = self.test_input.copy()
        invalid_input['num_classes'] = 'invalid'
        with self.assertRaises(Exception):
            model.predict_best_model(invalid_input)

    @patch('smartcal.meta_model.meta_model.MetaModel._load_component')
    def test_label_encoder_handling(self, mock_load):
        """Test prediction behavior with and without label encoder"""
        # Mock model and encoder
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.4, 0.3]])
        mock_model.classes_ = np.array(['model1', 'model2', 'model3'])
        
        # Test with label encoder
        mock_encoder = MagicMock()
        mock_encoder.classes_ = np.array(['model1', 'model2', 'model3'])
        mock_load.side_effect = [mock_model, mock_encoder]
        
        model = self.factory.create_model()
        predictions = model.predict_best_model(self.test_input)
        self.assertEqual(len(predictions), 3)
        
        # Test without label encoder (using model's classes_)
        mock_load.side_effect = [mock_model, None]
        model = self.factory.create_model()
        predictions = model.predict_best_model(self.test_input)
        self.assertEqual(len(predictions), 3)

    def test_model_prediction(self):
        """Test basic model prediction functionality"""
        # Create model
        model = self.factory.create_model()
        
        # Test prediction
        predictions = model.predict_best_model(self.test_input)
        
        # Verify predictions is a list
        self.assertIsInstance(predictions, list)
        self.assertTrue(len(predictions) > 0)
            
        # Verify probabilities sum to approximately 1
        self.assertAlmostEqual(sum(float(pred[1]) for pred in predictions), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
