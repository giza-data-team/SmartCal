import unittest
from unittest.mock import patch

from smartcal.meta_model.meta_model_factory import MetaModelFactory
from smartcal.meta_model.meta_model import MetaModel


class TestMetaModelFactory(unittest.TestCase):
    def setUp(self):
        self.factory = MetaModelFactory()
        self.test_input = {
            'num_classes': 5,
            'num_instances': 1000,
            'dataset_type': 'binary',
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

    @patch('smartcal.config.configuration_manager.configuration_manager.ConfigurationManager')
    def test_create_model_with_default_type(self, mock_config):
        # Setup mock config
        mock_config.return_value.meta_model_type = 'meta_model'
        
        # Create model without specifying type
        model = self.factory.create_model()
        
        # Verify it's a meta-model
        self.assertIsInstance(model, MetaModel)

    @patch('smartcal.config.configuration_manager.configuration_manager.ConfigurationManager')
    def test_create_model_with_specific_type(self, mock_config):
        # Setup mock config
        mock_config.return_value.meta_model_type = 'meta_model'
        
        # Create model with specific type
        model = self.factory.create_model('meta_model')
        
        # Verify it's a meta-model
        self.assertIsInstance(model, MetaModel)

    def test_create_model_with_invalid_type(self):
        # Verify it raises ValueError
        with self.assertRaises(ValueError):
            self.factory.create_model('invalid')

    def test_model_prediction(self):
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
