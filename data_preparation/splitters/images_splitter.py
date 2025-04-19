import torchvision.datasets as tv_datasets
import torch
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

from data_preparation.splitters.data_splitter_base import DatasetSplitter
from smartcal.utils.timer import time_operation


class ImagesSplitter(DatasetSplitter):
    def __init__(self, dataset_name, metadata_path, **kwargs):
        super().__init__(dataset_name=dataset_name, metadata_path=metadata_path, **kwargs)
        self.timing = {} 
        
        # Set Dataset as index if it's not already
        if 'Dataset' in self.metadata_df.columns:
            self.metadata_df.set_index('Dataset', inplace=True)
            
        if self.dataset_name not in self.metadata_df.index:
            raise ValueError(f"Dataset {dataset_name} not found in metadata file")
        
        self.dataset_config = self.metadata_df.loc[dataset_name]
        
        # Verify dataset type
        if self.dataset_config['Library'].lower() != 'torchvision':
            raise ValueError(f"Dataset {dataset_name} is not a torchvision dataset")

    def get_dataset_class(self):
        try:
            return getattr(tv_datasets, self.dataset_config['Torchvision_Name'])
        except AttributeError:
            self.logger.error(f"Dataset {self.dataset_config['Torchvision_Name']} not found in torchvision.datasets")
            raise

    def get_labels(self, dataset):
        if hasattr(dataset, 'labels'):
            return dataset.labels
        elif hasattr(dataset, 'targets'):
            return dataset.targets
        else:
            self.logger.error(f"Dataset {self.dataset_name} has no labels or targets attribute")
            raise AttributeError(f"Dataset {self.dataset_name} has no labels or targets attribute")
    
    def log_info(self, train_data, valid_data, test_data):
        if self.logs:
            self.logger.info(f"\nDataset: {self.dataset_name}")
            self.logger.info(f"Random Seed: {self.random_seed}")
            self.logger.info(f"Split Ratios (Train/Val): {self.split_ratios_images}")
            self.logger.info(f"Number of training samples: {len(train_data)}")
            self.logger.info(f"Number of validation samples: {len(valid_data)}")
            self.logger.info(f"Number of test samples: {len(test_data)}")

    @time_operation
    def split_dataset(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        DatasetClass = self.get_dataset_class()

        # Loading the training dataset and test set
        try:
            if self.dataset_name == 'SVHN':
                train_full = DatasetClass(root='./data', split='train', download=True)
                test_set = DatasetClass(root='./data', split='test', download=True)
            else:
                train_full = DatasetClass(root='./data', train=True, download=True)
                test_set = DatasetClass(root='./data', train=False, download=True)
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            raise

        labels = self.get_labels(train_full)

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=self.split_ratios_images[0],
            random_state=self.random_seed
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))

        train_set = Subset(train_full, train_idx)
        val_set = Subset(train_full, val_idx)

        if self.logs:
            self.log_info(train_set, val_set, test_set)

        return train_set, val_set, test_set
    
    def get_timing(self):
        return self.timing
    