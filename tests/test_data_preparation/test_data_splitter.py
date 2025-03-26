import unittest
import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import torch
from torch.utils.data import Dataset

from data_preparation.splitters.images_splitter import ImagesSplitter
from data_preparation.splitters.tabular_splitter import TabularSplitter
from data_preparation.splitters.language_splitter import LanguageSplitter

from config.configuration_manager.configuration_manager import ConfigurationManager

class MockDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)
        self.targets = torch.randint(0, 10, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TestLanguageSplitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_manager = ConfigurationManager()
        cls.metadata_path = cls.config_manager.config_language
        cls.logs = True  # Add this line to define logs at class level
        
        # Create mock IMDB data with column names matching your config file
        n_samples = 1000
        cls.imdb_data = pd.DataFrame({
            'review': [f'This is review {i}' for i in range(n_samples)],
            'sentiment': np.random.choice(['positive', 'negative'], size=n_samples)
        })
        
    def setUp(self):
        self.language_splitter = LanguageSplitter(
            dataset_name='IMDB',
            metadata_path=self.metadata_path,
            random_seed=13,
            logs=self.logs  # Use the class-level logs attribute
        )

    def test_split_dataset(self):
        # Debug logging
        if hasattr(self.language_splitter, 'logs') and self.language_splitter.logs:
            print(f"Mock data columns: {self.imdb_data.columns}")
            print(f"Target column from config: {self.language_splitter.target_cols[self.language_splitter.dataset_name]}")

        train_data, valid_data, test_data = self.language_splitter.split_dataset(self.imdb_data)
        
        # Test split sizes
        total_samples = len(self.imdb_data)
        self.assertAlmostEqual(len(train_data) / total_samples, 0.6, places=1)
        self.assertAlmostEqual(len(valid_data) / total_samples, 0.2, places=1)
        self.assertAlmostEqual(len(test_data) / total_samples, 0.2, places=1)
        
        # Additional tests
        self.assertEqual(list(train_data.columns), list(self.imdb_data.columns))
        self.assertTrue('sentiment' in train_data.columns)
        
class TestImagesSplitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_manager = ConfigurationManager()
        cls.metadata_path = cls.config_manager.config_img
        cls.splits_img = cls.config_manager.split_ratios_images
        
        # Create mock image dataset
        cls.mock_dataset = MockDataset(1000)
        
    def setUp(self):
        self.image_splitter = ImagesSplitter(
            'SVHN',
            split_ratios_images=self.splits_img,
            metadata_path=self.metadata_path,
            logs=True
        )

    @patch('torchvision.datasets.SVHN')
    def test_split_dataset(self, mock_svhn):
        # Configure mock
        mock_svhn.return_value = self.mock_dataset
        
        train_set, val_set, test_set = self.image_splitter.split_dataset()
        
        # Test that we get the correct types
        self.assertTrue(hasattr(train_set, '__len__'))
        self.assertTrue(hasattr(val_set, '__len__'))
        self.assertTrue(hasattr(test_set, '__len__'))
        
        # Test split ratios
        total_train_val = len(train_set) + len(val_set)
        self.assertAlmostEqual(len(train_set) / total_train_val, 0.75, places=1)
        self.assertAlmostEqual(len(val_set) / total_train_val, 0.25, places=1)

class TestTabularSplitter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_manager = ConfigurationManager()
        cls.metadata_path = cls.config_manager.config_tabular
        cls.logs = True
        
        # Create mock iris data matching your config file's target column name
        n_samples = 150
        cls.data = pd.DataFrame({
            'sepal_length': np.random.normal(5.5, 1, n_samples),
            'sepal_width': np.random.normal(3.5, 0.5, n_samples),
            'petal_length': np.random.normal(4.5, 1, n_samples),
            'petal_width': np.random.normal(1.5, 0.5, n_samples),
            'Species': np.random.choice(['setosa', 'versicolor', 'virginica'], size=n_samples)  # Make sure this matches your config
        })
        
    def setUp(self):
        self.tabular_splitter = TabularSplitter(
            dataset_name='Iris',  # Make sure this matches your config file
            metadata_path=self.metadata_path,
            random_seed=42,
            logs=self.logs
        )

    def test_split_dataset(self):
        train_data, valid_data, test_data = self.tabular_splitter.split_dataset(self.data)
        
        # Test split sizes
        total_samples = len(self.data)
        self.assertAlmostEqual(len(train_data) / total_samples, 0.6, places=1)
        self.assertAlmostEqual(len(valid_data) / total_samples, 0.2, places=1)
        self.assertAlmostEqual(len(test_data) / total_samples, 0.2, places=1)
        
        # Test data integrity
        self.assertEqual(list(train_data.columns), list(self.data.columns))
        target_col = self.tabular_splitter.target_cols[self.tabular_splitter.dataset_name]
        self.assertTrue(target_col in train_data.columns)
        
        # Test that no data was lost
        self.assertEqual(
            len(train_data) + len(valid_data) + len(test_data),
            len(self.data)
        )
        
    def test_invalid_dataset(self):
        """Test that an invalid dataset name raises an error"""
        with self.assertRaises(ValueError) as context:
            TabularSplitter(
                dataset_name='invalid_dataset',
                metadata_path=self.metadata_path,
                logs=self.logs
            )
        
        error_message = str(context.exception)
        self.assertTrue("not found in metadata file" in error_message)
        self.assertTrue("Available datasets" in error_message)

    def test_missing_target_column(self):
        """Test that missing target column raises an error"""
        # Create data without target column
        bad_data = self.data.drop(columns=[self.tabular_splitter.target_cols[self.tabular_splitter.dataset_name]])
        
        with self.assertRaises(KeyError):
            self.tabular_splitter.split_dataset(bad_data)
