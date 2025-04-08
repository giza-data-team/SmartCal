import unittest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from torch.utils.data import Dataset, DataLoader
import shutil
from pathlib import Path
from Package.src.SmartCal.config.enums.language_models_enum import ModelType
from data_preparation.splitters.tabular_splitter import TabularSplitter
from data_preparation.preprocessors.images_preprocessor import ImagePreprocessor
from data_preparation.preprocessors.language_preprocessor import LanguagePreprocessor
from data_preparation.preprocessors.tabular_preprocessor import TabularPreprocessor

from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from Package.src.SmartCal.config.enums.language_models_enum import LanguageModelsEnum

class MockDataset(Dataset):
    """Mock dataset for image testing"""
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)
        self.targets = torch.randint(0, 10, (size,))
        self.labels = self.targets 

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TestPreprocessors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize configuration manager
        cls.config_manager = ConfigurationManager()
        
        # Create mock metadata for tabular
        cls.mock_tabular_metadata = pd.DataFrame({
            'Dataset': ['mock_tabular'],
            'Target': ['target'],
            'Type': ['Classification'],
            'Task': ['Binary Classification']
        })
        
        # Create mock metadata for language
        cls.mock_language_metadata = pd.DataFrame({
            'Dataset': ['mock_language'],
            'Target': ['label'],
            'Text': ['text'],
            'Type': ['Classification'],
            'Task': ['Multi Classification']
        })
        
        # Create mock metadata for images
        cls.mock_image_metadata = pd.DataFrame({
            'Dataset': ['CIFAR100'],
            'Torchvision_Name': ['CIFAR100'],
            'Mean': ['[0.5, 0.5, 0.5]'],
            'STD': ['[0.5, 0.5, 0.5]'],
            'Type': ['torchvision']
        })

    def setUp(self):
        """Set up test environment before each test"""
        self.random_seed = 42
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        # Create temporary directory for test files
        self.temp_dir = Path("temp_test_files")
        self.temp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up after each test"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_mock_tabular_data(self, size=1000):
        """Create mock tabular data with various column types"""
        data = pd.DataFrame({
            # Numerical columns
            'num_normal': np.random.normal(0, 1, size),
            'num_uniform': np.random.uniform(0, 100, size),
            
            # Categorical columns
            'cat_low': np.random.choice(['A', 'B', 'C'], size),
            'cat_high': [f'val_{i%20}' for i in range(size)],
            
            # Integer column
            'int_col': np.random.randint(0, 100, size),
            
            # Target column
            'target': np.random.randint(0, 2, size)
        })
        
        # Add missing values (except in target)
        for col in data.columns:
            if col != 'target':
                mask = np.random.random(size) < 0.1
                data.loc[mask, col] = np.nan
                
        return data

    def create_mock_language_data(self, size=1000):
        """Create mock language data with realistic text"""
        words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
        texts = []
        for _ in range(size):
            length = np.random.randint(5, 15)
            text = ' '.join(np.random.choice(words, size=length))
            texts.append(text)
            
        return pd.DataFrame({
            'text': texts,
            'label': np.random.randint(0, 4, size)
        })

    def validate_preprocessor_output(self, X, y, expected_samples, expected_features=None):
        """Utility method to validate preprocessor outputs"""
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), expected_samples)
        self.assertEqual(len(y), expected_samples)
        if expected_features:
            self.assertEqual(X.shape[1], expected_features)

    @patch('pandas.read_excel')
    def test_tabular_preprocessor(self, mock_read_excel):
        """Test tabular data preprocessing"""
        try:
            # Setup mock
            mock_read_excel.return_value = self.mock_tabular_metadata
            
            # Create and split mock data
            mock_data = self.create_mock_tabular_data()
            
            # Initialize splitter
            splitter = TabularSplitter(
                dataset_name='mock_tabular',
                metadata_path="mock_path",
                logs=True
            )
            
            # Split data
            train_data, valid_data, test_data = splitter.split_dataset(mock_data)

            # Test preprocessor
            preprocessor = TabularPreprocessor(
                dataset_name='mock_tabular',
                metadata_path="mock_path",
                logs=True
            )

            # Process data
            X_train, y_train = preprocessor.fit_transform(train_data)
            X_valid, y_valid = preprocessor.transform(valid_data)
            X_test, y_test = preprocessor.transform(test_data)

            # Validate outputs
            self.validate_preprocessor_output(X_train, y_train, len(train_data))
            self.validate_preprocessor_output(X_valid, y_valid, len(valid_data))
            self.validate_preprocessor_output(X_test, y_test, len(test_data))

            # Additional assertions
            self.assertTrue(all(X_train.dtypes != 'object'))
            self.assertTrue(X_train.isnull().sum().sum() == 0)
            self.assertEqual(X_train.shape[1], X_valid.shape[1])
            
        except Exception as e:
            self.fail(f"Tabular preprocessor test failed: {str(e)}")
    
    @patch('pandas.read_excel')
    def test_language_preprocessor(self, mock_read_excel):
        """Test language data preprocessing"""
        try:
            # Setup mock
            mock_read_excel.return_value = self.mock_language_metadata
            
            # Create mock data
            mock_data = self.create_mock_language_data()
            
            # Test FastText preprocessor
            preprocessor_fasttext = LanguagePreprocessor(
                model_name=ModelType.WordEmbeddingModel,
                dataset_name='mock_language',
                metadata_path="mock_path",
                logs=True
            )

            train_texts, train_labels = preprocessor_fasttext.fit_transform(mock_data)
            
            # FastText specific assertions
            self.assertIsInstance(train_texts, (list, pd.Series))
            self.assertIsInstance(train_labels, list)
            # Check if the FastText format is correct
            self.assertTrue(all(isinstance(text, str) for text in train_texts))
            self.assertTrue(all(label.startswith('__label__') for label in train_labels))
            
            # Test transform method for FastText
            test_texts, test_labels = preprocessor_fasttext.transform(mock_data)
            self.assertIsInstance(test_texts, pd.Series)
            self.assertIsInstance(test_labels, list)
            self.assertTrue(all(isinstance(text, str) for text in test_texts))
            self.assertTrue(all(label.startswith('__label__') for label in test_labels))
            
            # Test BERT preprocessor
            preprocessor_bert = LanguagePreprocessor(
                model_name=ModelType.BERT,
                dataset_name='mock_language',
                metadata_path="mock_path",
                logs=True
            )

            train_inputs, train_labels = preprocessor_bert.fit_transform(mock_data)
            
            # BERT specific assertions
            self.assertIn('input_ids', train_inputs)
            self.assertIn('attention_mask', train_inputs)
            self.assertIsInstance(train_labels, np.ndarray)
            
            # Test transform method for BERT
            test_inputs, test_labels = preprocessor_bert.transform(mock_data)
            self.assertIn('input_ids', test_inputs)
            self.assertIn('attention_mask', test_inputs)
            self.assertIsInstance(test_labels, np.ndarray)
            
        except Exception as e:
            self.fail(f"Language preprocessor test failed: {str(e)}")

    @patch('pandas.read_excel')
    def test_image_preprocessor(self, mock_read_excel):
        """Test image data preprocessing"""
        try:
            # Setup mock
            mock_read_excel.return_value = self.mock_image_metadata
            
            # Create mock datasets
            train_dataset = MockDataset(size=100)
            val_dataset = MockDataset(size=20)
            test_dataset = MockDataset(size=20)

            # Test preprocessor
            preprocessor = ImagePreprocessor(
                dataset_name='CIFAR100',
                metadata_path="mock_path",
                logs=True
            )

            # Process datasets
            train_loader = preprocessor.fit_transform(train_dataset)
            val_loader = preprocessor.transform(val_dataset)
            test_loader = preprocessor.transform(test_dataset)

            # Get a batch
            images, labels = next(iter(train_loader))

            # Image-specific assertions
            self.assertEqual(images.shape[1], 3)  # channels
            self.assertEqual(images.shape[2], 224)  # height
            self.assertEqual(images.shape[3], 224)  # width
            self.assertIsInstance(images, torch.Tensor)
            self.assertIsInstance(labels, torch.Tensor)
            self.assertIsInstance(val_loader, DataLoader)
            self.assertIsInstance(test_loader, DataLoader)
            
        except Exception as e:
            self.fail(f"Image preprocessor test failed: {str(e)}")