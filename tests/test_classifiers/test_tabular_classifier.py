import unittest
import tempfile
import shutil
import os
import numpy as np
import pandas as pd
from unittest.mock import patch

from classifiers.tabular_classifier import TabularClassifier
from Package.src.SmartCal.config.enums.tabular_models_enum import TabularModelsEnum


class TestTabularClassifier(unittest.TestCase):
    """ Unit tests for the TabularClassifier class """

    @classmethod
    def setUpClass(cls):
        """ Set up resources shared across all test cases """
        cls.temp_dir = tempfile.mkdtemp()

        # Generate a larger dataset
        num_train_samples = 100  # 100 training samples
        num_test_samples = 20  # 20 test samples
        num_features = 5         # 5 features

        # Create random feature data
        cls.X_train = pd.DataFrame(np.random.rand(num_train_samples, num_features), 
                            columns=[f"feature_{i}" for i in range(num_features)])
        cls.X_test = pd.DataFrame(np.random.rand(num_test_samples, num_features), 
                            columns=[f"feature_{i}" for i in range(num_features)])

        # Create random binary target labels (0 or 1)
        cls.y_train = pd.Series(np.random.randint(0, 2, num_train_samples), name="target")
        cls.y_test = pd.Series(np.random.randint(0, 2, num_test_samples), name="target")

        cls.classifier = TabularClassifier(TabularModelsEnum.NAIVE_BAYES, seed=42)

    @classmethod
    def tearDownClass(cls):
        """ Clean up resources after all test cases """
        shutil.rmtree(cls.temp_dir)

    def test_train_valid_data(self):
        """ Test training process with valid input data """
        self.classifier.train(self.X_train, self.y_train)
        self.assertIn("accuracy", self.classifier.training_metrics)
        self.assertIn("loss", self.classifier.training_metrics)

    def test_train_with_minimal_data(self):
        """ Test if training works with a minimal dataset (2+ samples with different classes) """
        X_small = pd.DataFrame([[0.5, 0.3, 0.7, 0.9, 0.2], [0.1, 0.8, 0.4, 0.6, 0.7]], 
                            columns=[f"feature_{i}" for i in range(5)])
        y_small = pd.Series([1, 0], name="target")  # Ensure at least two classes

        try:
            self.classifier.train(X_small, y_small)
        except Exception as e:
            self.fail(f"Training with minimal data failed: {e}")

    def test_train_high_dimensional_data(self):
        """ Test training with a high-dimensional dataset (10,000 features) """
        X_high_dim = pd.DataFrame(np.random.rand(100, 10000), columns=[f"feature_{i}" for i in range(10000)])
        y_high_dim = pd.Series(np.random.randint(0, 2, 100), name="target")

        self.classifier.train(X_high_dim, y_high_dim)
        self.assertIn("accuracy", self.classifier.training_metrics)
        self.assertIn("loss", self.classifier.training_metrics)
    
    def test_train_multiclass_classification(self):
        """ Ensure the classifier can handle multi-class datasets properly """
        X_multi = self.X_train
        y_multi = pd.Series(np.random.randint(0, 3, len(self.X_train)), name="target")  # 3-class problem

        self.classifier.train(X_multi, y_multi)
        self.assertIn("accuracy", self.classifier.training_metrics)
        self.assertIn("loss", self.classifier.training_metrics)

        predictions = self.classifier.predict(self.X_test)
        self.assertTrue(set(predictions).issubset({0, 1, 2}))  # Ensure predictions belong to valid classes

    def test_training_metrics_values(self):
        """ Test if training metrics are recorded and valid. """
        self.classifier.train(self.X_train, self.y_train)
        expected_metrics = self.classifier.training_metrics.keys()
        for metric in expected_metrics:
            self.assertIsNotNone(self.classifier.training_metrics[metric])

    def test_training_time_calculation(self):
        """ Test if training time is properly recorded. """
        self.classifier.train(self.X_train, self.y_train)
        self.assertGreater(self.classifier.training_time, 0)  # Training time should be greater than 0

    def test_train_empty_dataset(self):
        """ Ensure that training on an empty dataset raises an error. """
        empty_X_train = pd.DataFrame()
        empty_y_train = pd.Series()

        with self.assertRaises(RuntimeError):
            self.classifier.train(empty_X_train, empty_y_train)

    def test_train_invalid_data(self):
        """ Test training with invalid data types (should raise RuntimeError) """
        with self.assertRaises(RuntimeError):  
            self.classifier.train("invalid_data", self.y_train)

        with self.assertRaises(RuntimeError):  
            self.classifier.train(self.X_train, "invalid_labels")

    def test_train_mismatched_shapes(self):
        """ Test training with mismatched feature and label counts (should raise RuntimeError) """
        y_mismatched = self.y_train.iloc[:-1]
        with self.assertRaises(RuntimeError):  
            self.classifier.train(self.X_train, y_mismatched)

    def test_predict_valid_data(self):
        """ Test the predict method and ensure output shape is correct """
        self.classifier.train(self.X_train, self.y_train)
        predictions = self.classifier.predict(self.X_test, self.y_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_prediction_time_calculation(self):
        """ Test if prediction time is properly recorded. """
        self.classifier.train(self.X_train, self.y_train)
        self.classifier.predict(self.X_test, self.y_test)
        self.classifier.predict_prob(self.X_test)
        
        # Check for correct attribute names
        self.assertGreater(self.classifier.testing_time_predict, 0)  # Fixed to use `testing_time_predict`
        self.assertGreater(self.classifier.testing_time_predictprob, 0)  # Fixed to use `testing_time_predictprob`

    def test_testing_metrics_values(self):
        """ Test if testing metrics are recorded and valid. """
        self.classifier.train(self.X_train, self.y_train)
        self.classifier.predict(self.X_test, self.y_test)
        expected_metrics = self.classifier.testing_metrics.keys()
        for metric in expected_metrics:
            self.assertIsNotNone(self.classifier.testing_metrics[metric])

    def test_predict_invalid_data(self):
        """ Test prediction with invalid feature types (should raise RuntimeError) """
        with self.assertRaises(RuntimeError):  
            self.classifier.predict("invalid_data", self.y_test)

    def test_predict_mismatched_shapes(self):
        """ Test prediction with mismatched feature and label counts (should raise RuntimeError) """
        y_mismatched = self.y_test.iloc[:-1]
        with self.assertRaises(RuntimeError):  
            self.classifier.predict(self.X_test, y_mismatched)

    def test_predict_prob_valid_data(self):
        """ Test predict_prob method returns probability distributions """
        self.classifier.train(self.X_train, self.y_train)
        probabilities = self.classifier.predict_prob(self.X_test)

        self.assertEqual(len(probabilities), len(self.X_test))  # Should match input size
        self.assertIsInstance(probabilities, np.ndarray)  # Should return a list of probabilities
        self.assertTrue(all(isinstance(p, np.ndarray) for p in probabilities))  # Each item should be a list (probability vector)

    def test_predict_prob_invalid_data(self):
        """ Test predict_prob with invalid feature types (should raise RuntimeError) """
        with self.assertRaises(RuntimeError):  
            self.classifier.predict_prob("invalid_data")

    def test_predict_prob_output_shape(self):
        """ Ensure predict_prob returns an array of shape (n_samples, n_classes) """
        self.classifier.train(self.X_train, self.y_train)
        probabilities = self.classifier.predict_prob(self.X_test)

        self.assertEqual(len(probabilities), len(self.X_test))  # Should match number of samples
        self.assertEqual(len(probabilities[0]), len(set(self.y_train)), "Each sample should have a probability vector matching the number of classes")

    def test_log_results(self):
        """ Test the log_results function to ensure it returns the expected structure. """
        self.classifier.train(self.X_train, self.y_train)
        self.classifier.predict(self.X_test, self.y_test)
        self.classifier.predict_prob(self.X_test)
        log_data = self.classifier.log_results()
        expected_keys = [
            "model", "dataset_type", "device", "training_metrics",
            "testing_metrics", "training_time", "testing_time_predict",
            "testing_time_predictprob"
        ]
        for key in expected_keys:
            self.assertIn(key, log_data)
        self.assertIsInstance(log_data["training_metrics"], dict)
        self.assertIsInstance(log_data["testing_metrics"], dict)

    def test_load_model_without_prior_training(self):
        """ Ensure loading a model from a non-existent file raises FileNotFoundError """
        untrained_classifier = TabularClassifier(TabularModelsEnum.RANDOM_FOREST, seed=42)

        with self.assertRaises(FileNotFoundError):  # Expect FileNotFoundError instead of RuntimeError
            untrained_classifier.load_model("non_existent_model.pkl")

    def test_save_and_load_model(self):
        """ Test saving and loading the model correctly """
        model_path = os.path.join(self.temp_dir, "tabular_model.pkl")
        
        self.classifier.train(self.X_train, self.y_train)
        self.classifier.save_model(model_path)
        self.assertTrue(os.path.exists(model_path))  # Check if model file exists

        # Load model and verify predictions remain consistent
        new_classifier = TabularClassifier(TabularModelsEnum.RANDOM_FOREST, seed=42)
        new_classifier.load_model(model_path)
        predictions_after_loading = new_classifier.predict(self.X_test, self.y_test)
        
        self.assertEqual(len(predictions_after_loading), len(self.X_test))  # Check shape consistency

    def test_probability_normalization_specific_values(self):
        """Test if the classifier correctly normalizes specific probability values."""
        with patch('sklearn.naive_bayes.GaussianNB') as mock_nb:
            instance = mock_nb.return_value
            instance.predict.return_value = np.array([0, 1])
            
            # Define specific probability values to test
            # These values don't sum to 1 (they sum to approx. 1.00000035)
            unnormalized_probs = np.array([
                [0.03477070853114128, 0.9652292728424072]
            ])
            instance.predict_proba.return_value = unnormalized_probs
            classifier = TabularClassifier(TabularModelsEnum.NAIVE_BAYES, seed=42)
            classifier.model = instance  # Replace with mock
            
            # Test predict_prob method
            actual_probs = classifier.predict_prob(self.X_test[:1])  # Use just first test sample
            
            # Verify that the probabilities sum to 1
            row_sum = np.sum(actual_probs[0])
            np.testing.assert_almost_equal(row_sum, 1.0, decimal=8)
            
            # Check that the values are normalized proportionally
            original_ratio = unnormalized_probs[0][1] / unnormalized_probs[0][0]
            normalized_ratio = actual_probs[0][1] / actual_probs[0][0]
            np.testing.assert_almost_equal(original_ratio, normalized_ratio, decimal=8)
            
    def test_save_model_invalid_path(self):
        """ Test saving the model to an invalid path (should raise RuntimeError) """
        with self.assertRaises(RuntimeError):  
            self.classifier.save_model(None)

    def test_load_model_invalid_path(self):
        """ Test loading a model from a non-existent file (should raise FileNotFoundError) """
        with self.assertRaises(FileNotFoundError):
            self.classifier.load_model("non_existent_model.pkl")

    def test_model_consistency_after_save_load(self):
        """ Ensure that a model makes the same predictions before and after saving/loading. """
        model_path = os.path.join(self.temp_dir, "saved_model.pkl")

        # Train the model and make initial predictions
        self.classifier.train(self.X_train, self.y_train)
        initial_predictions = self.classifier.predict(self.X_test, self.y_test)

        # Save and reload the model
        self.classifier.save_model(model_path)
        self.classifier.load_model(model_path)

        # Make predictions again and compare
        loaded_predictions = self.classifier.predict(self.X_test, self.y_test)
        np.testing.assert_array_equal(initial_predictions, loaded_predictions, "Predictions should match after save/load.")

if __name__ == "__main__":
    unittest.main()
