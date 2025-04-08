import unittest
import tempfile
import shutil
import torch

from Package.src.SmartCal.config.enums.language_models_enum import LanguageModelsEnum
from classifiers.language_classifier import LanguageClassifier


class TestLanguageClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up test data for all test cases."""
        cls.train_texts_fasttext = [
            "__label__2 This is a neutral review",
            "__label__1 This is a positive review",
            "__label__0 This is a negative review"
        ]
        cls.train_labels_fasttext = ["__label__2", "__label__1", "__label__0"]

        cls.val_texts_fasttext = ["This review is excellent", "This review is average", "This review is terrible"]
        cls.val_labels_fasttext = ["__label__2", "__label__1", "__label__0"]

        cls.test_texts_fasttext = ["This review is excellent", "This review is average", "This review is terrible"]
        cls.test_labels_fasttext = ["__label__2", "__label__1", "__label__0"]
        
        cls.train_inputs_transformer = {
            "input_ids": [[101, 2023, 2003, 1037, 3893, 3315, 102], 
                          [101, 2023, 2003, 1037, 4997, 3315, 102], 
                          [101, 2023, 2003, 1037, 3793, 3315, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1, 1], 
                               [1, 1, 1, 1, 1, 1, 1], 
                               [1, 1, 1, 1, 1, 1, 1]]
        }
        cls.train_labels_transformer = [2, 1, 0]

        cls.val_inputs_transformer = {
            "input_ids": [
                [101, 2023, 3315, 2003, 6581, 102],  
                [101, 2023, 3315, 2003, 5053, 102],  
                [101, 2023, 3315, 2003, 6659, 102]   
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]
            ]
        }
        cls.val_labels_transformer = [2, 1, 0]

        cls.test_inputs_transformer = {
            "input_ids": [
                [101, 2023, 3315, 2003, 6581, 102],  
                [101, 2023, 3315, 2003, 5053, 102],  
                [101, 2023, 3315, 2003, 6659, 102]   
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]
            ]
        }
        cls.test_labels_transformer = [2, 1, 0]

        cls.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def setUp(self):
        """Create temporary directories for model saving and loading."""
        self.temp_dir_fasttext = tempfile.mkdtemp()
        self.temp_dir_transformer = tempfile.mkdtemp()

    def tearDown(self):
        """Remove temporary directories after each test."""
        shutil.rmtree(self.temp_dir_fasttext)
        shutil.rmtree(self.temp_dir_transformer)

    def test_train_fasttext(self):
        """Test training FastText model with validation data."""
        fasttext_classifier = LanguageClassifier(LanguageModelsEnum.FASTTEXT, device=self.DEVICE)
        fasttext_classifier.train(self.train_texts_fasttext, self.train_labels_fasttext, 
                                  self.val_texts_fasttext, self.val_labels_fasttext)
        self.assertIsNotNone(fasttext_classifier.model)
        self.assertIn("loss", fasttext_classifier.training_metrics)
        self.assertIn("accuracy", fasttext_classifier.training_metrics)

    def test_train_transformer(self):
        """Test training Transformer model with validation data."""
        transformer_classifier = LanguageClassifier(LanguageModelsEnum.TINYBERT, device=self.DEVICE)
        transformer_classifier.train(self.train_inputs_transformer, self.train_labels_transformer, 
                                     self.val_inputs_transformer, self.val_labels_transformer)
        self.assertIsNotNone(transformer_classifier.model)
        self.assertIn("loss", transformer_classifier.training_metrics)
        self.assertIn("accuracy", transformer_classifier.training_metrics)

    def test_predict_fasttext(self):
        """Test prediction with FastText model."""
        fasttext_classifier = LanguageClassifier(LanguageModelsEnum.FASTTEXT, device=self.DEVICE)
        fasttext_classifier.train(self.train_texts_fasttext, self.train_labels_fasttext, 
                                  self.val_texts_fasttext, self.val_labels_fasttext)
        predictions = fasttext_classifier.predict(self.test_texts_fasttext)
        self.assertEqual(len(predictions), len(self.test_texts_fasttext))

    def test_predict_prob_fasttext(self):
        """Test probability prediction with FastText model."""
        fasttext_classifier = LanguageClassifier(LanguageModelsEnum.FASTTEXT, device=self.DEVICE)
        fasttext_classifier.train(self.train_texts_fasttext, self.train_labels_fasttext, 
                                  self.val_texts_fasttext, self.val_labels_fasttext)
        fasttext_classifier.predict(self.test_texts_fasttext, self.test_labels_fasttext)
        probabilities = fasttext_classifier.predict_prob(self.test_texts_fasttext)
        self.assertEqual(len(probabilities), len(self.test_texts_fasttext))

    def test_predict_transformer(self):
        """Test prediction with Transformer model."""
        transformer_classifier = LanguageClassifier(LanguageModelsEnum.TINYBERT, device=self.DEVICE)
        transformer_classifier.train(self.train_inputs_transformer, self.train_labels_transformer, 
                                     self.val_inputs_transformer, self.val_labels_transformer)
        predictions = transformer_classifier.predict(self.val_inputs_transformer)
        self.assertEqual(len(predictions), len(self.val_inputs_transformer["input_ids"]))

    def test_predict_prob_transformer(self):
        """Test probability prediction with Transformer model."""
        transformer_classifier = LanguageClassifier(LanguageModelsEnum.TINYBERT, device=self.DEVICE)
        transformer_classifier.train(self.train_inputs_transformer, self.train_labels_transformer, 
                                     self.val_inputs_transformer, self.val_labels_transformer)
        predictions = transformer_classifier.predict(self.val_inputs_transformer, self.val_labels_transformer)
        probabilities = transformer_classifier.predict_prob(self.val_inputs_transformer)
        self.assertEqual(len(probabilities), len(self.val_inputs_transformer["input_ids"]))

    def test_log_results_fasttext(self):
        """Test log_results method for FastText classifier."""
        fasttext_classifier = LanguageClassifier(LanguageModelsEnum.FASTTEXT, device=self.DEVICE)
        fasttext_classifier.train(self.train_texts_fasttext, self.train_labels_fasttext, self.val_texts_fasttext, self.val_labels_fasttext)
        fasttext_classifier.predict(self.test_texts_fasttext, self.test_labels_fasttext)
        fasttext_classifier.predict_prob(self.test_texts_fasttext)
        log_output = fasttext_classifier.log_results()
        print(log_output)

        self.assertIsInstance(log_output, dict, "log_results should return a dictionary.")
        self.assertIn("model", log_output)
        self.assertIn("training_metrics", log_output)
        self.assertIn("testing_metrics", log_output)
        self.assertIn("predicted_labels", log_output)

    def test_log_results_transformer(self):
        """Test log_results method for Transformer classifier."""
        transformer_classifier = LanguageClassifier(LanguageModelsEnum.TINYBERT, device=self.DEVICE)
        transformer_classifier.train(self.train_inputs_transformer, self.train_labels_transformer, self.val_inputs_transformer, self.val_labels_transformer)
        transformer_classifier.predict(self.test_inputs_transformer, self.test_labels_transformer)
        transformer_classifier.predict_prob(self.test_inputs_transformer)
        log_output = transformer_classifier.log_results()
        print(log_output)

        self.assertIsInstance(log_output, dict, "log_results should return a dictionary.")
        self.assertIn("model", log_output)
        self.assertIn("training_metrics", log_output)
        self.assertIn("testing_metrics", log_output)
        self.assertIn("predicted_labels", log_output)

    def test_save_and_load_fasttext(self):
        """Test saving and loading FastText model."""
        fasttext_classifier = LanguageClassifier(LanguageModelsEnum.FASTTEXT, device=self.DEVICE)
        fasttext_classifier.train(self.train_texts_fasttext, self.train_labels_fasttext, 
                                  self.val_texts_fasttext, self.val_labels_fasttext)
        fasttext_classifier.save_model(self.temp_dir_fasttext)
        loaded_classifier = LanguageClassifier(LanguageModelsEnum.FASTTEXT, DEVICE=self.DEVICE)
        loaded_classifier.load_model(self.temp_dir_fasttext)
        self.assertIsNotNone(loaded_classifier.model)

    def test_save_and_load_transformer(self):
        """Test saving and loading Transformer model."""
        transformer_classifier = LanguageClassifier(LanguageModelsEnum.TINYBERT, device=self.DEVICE)
        transformer_classifier.train(self.train_inputs_transformer, self.train_labels_transformer, 
                                     self.val_inputs_transformer, self.val_labels_transformer)
        transformer_classifier.save_model(self.temp_dir_transformer)
        loaded_classifier = LanguageClassifier(LanguageModelsEnum.TINYBERT, device=self.DEVICE)
        loaded_classifier.load_model(self.temp_dir_transformer)
        self.assertIsNotNone(loaded_classifier.model)

if __name__ == "__main__":
    unittest.main()
