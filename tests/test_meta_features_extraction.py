import unittest
import numpy as np
import os

from smartcal.meta_features_extraction.meta_features_extraction import MetaFeaturesExtractor


class TestMetaFeaturesExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MetaFeaturesExtractor()
        self.y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])  # Sample ground truth labels
        self.y_pred_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6],
            [0.3, 0.6, 0.1],
            [0.7, 0.2, 0.1],
            [0.1, 0.4, 0.5],
            [0.3, 0.5, 0.2],
            [0.9, 0.05, 0.05]
        ])

    def test_process_features(self):
        features = self.extractor.process_features(self.y_true, self.y_pred_prob)
        expected_features = [
            'num_classes', 'num_instances', 'dataset_type', 'class_imbalance_ratio',
            'actual_predictions_entropy', 'Confidence_Mean', 'Confidence_Median',
            'Confidence_Std', 'Confidence_Var', 'Confidence_Entropy',
            'Confidence_Skewness', 'Confidence_Kurtosis', 'Confidence_Min', 'Confidence_Max',
            'Classification_Accuracy', 'Classification_Precision_Micro', 'Classification_Precision_Macro',
            'Classification_Precision_Weighted', 'Classification_Recall_Micro',
            'Classification_Recall_Macro', 'Classification_Recall_Weighted',
            'Classification_F1_Micro', 'Classification_F1_Macro', 'Classification_F1_Weighted',
            'ECE_before', 'MCE_before', 'ConfECE_before', 'Wasserstein_Mean', 'Wasserstein_Median',
            'Wasserstein_Std', 'Wasserstein_Var', 'Wasserstein_Entropy', 'Wasserstein_Skewness',
            'Wasserstein_Kurtosis', 'Wasserstein_Min', 'Wasserstein_Max', 'KL_Divergence_Mean',
            'KL_Divergence_Median', 'KL_Divergence_Std', 'KL_Divergence_Var', 'KL_Divergence_Entropy',
            'KL_Divergence_Skewness', 'KL_Divergence_Kurtosis', 'KL_Divergence_Min', 'KL_Divergence_Max',
            'Jensen_Shannon_Mean', 'Jensen_Shannon_Median', 'Jensen_Shannon_Std', 'Jensen_Shannon_Var',
            'Jensen_Shannon_Entropy', 'Jensen_Shannon_Skewness', 'Jensen_Shannon_Kurtosis',
            'Jensen_Shannon_Min', 'Jensen_Shannon_Max', 'Bhattacharyya_Mean', 'Bhattacharyya_Median',
            'Bhattacharyya_Std', 'Bhattacharyya_Var', 'Bhattacharyya_Entropy', 'Bhattacharyya_Skewness',
            'Bhattacharyya_Kurtosis', 'Bhattacharyya_Min', 'Bhattacharyya_Max'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features, f"Missing feature: {feature}")
        
        self.assertIsInstance(features['num_classes'], int)
        self.assertGreaterEqual(features['num_instances'], 1)
        self.assertLessEqual(features['Classification_Accuracy'], 1.0)

    def test_handle_empty_data(self):
        empty_y_true = np.array([])
        empty_y_pred_prob = np.array([])
        with self.assertRaises(Exception):
            self.extractor.process_features(empty_y_true, empty_y_pred_prob)

    def test_save_to_csv(self):
        csv_filename = "test_meta_features.csv"
        if os.path.exists(csv_filename):
            os.remove(csv_filename)
        
        self.extractor.process_features(self.y_true, self.y_pred_prob)
        self.extractor.save_to_csv(csv_filename)
        
        self.assertTrue(os.path.exists(csv_filename))
        os.remove(csv_filename)

if __name__ == '__main__':
    unittest.main()
