from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
import logging
import pandas as pd

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config_manager = ConfigurationManager()

class DatasetSplitter(ABC):
    def __init__(self, dataset_name, metadata_path, random_seed=config_manager.random_seed,
                 split_ratios=config_manager.split_ratios,
                 split_ratios_images=config_manager.split_ratios_images, logs=False):
        
        self.dataset_name = dataset_name
        self.metadata_path = metadata_path
        self.logs = logs
        
        self.random_seed = random_seed
        self.split_ratios = split_ratios
        self.split_ratios_images = split_ratios_images

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        # Load and validate metadata
        self.load_and_validate_metadata()

    def load_and_validate_metadata(self):
        """Load metadata file and validate dataset existence"""
        try:
            # Load metadata file
            self.metadata_df = pd.read_excel(self.metadata_path)
            
            # Validate dataset exists in metadata
            if self.dataset_name not in self.metadata_df['Dataset'].values:
                available_datasets = list(self.metadata_df['Dataset'].values)
                raise ValueError(
                    f"Dataset '{self.dataset_name}' not found in metadata file. "
                    f"Available datasets: {available_datasets}"
                )
            
            # Create target_cols dictionary if 'Target' column exists
            if 'Target' in self.metadata_df.columns:
                self.target_cols = dict(zip(self.metadata_df['Dataset'], self.metadata_df['Target']))
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file '{self.metadata_path}' not found")
        except ValueError as e:
            # Re-raise ValueError without wrapping it in another exception
            raise e
        except Exception as e:
            raise Exception(f"Error reading metadata file: {str(e)}")
        
    def log_info(self, train_data, valid_data, test_data):
        if self.logs:
            self.logger.info(f"\nDataset: {self.dataset_name}")
            self.logger.info(f"Random Seed: {self.random_seed}")
            self.logger.info(f"Split Ratios (Train/Val/Test): {self.split_ratios}")
            self.logger.info(f"Number of training samples: {len(train_data)}")
            self.logger.info(f"Number of validation samples: {len(valid_data)}")
            self.logger.info(f"Number of test samples: {len(test_data)}")

    def split_structured_data(self, data, target_col):
        # Drop rows where target column has NaN values
        original_length = len(data)
        data = data.dropna(subset=[target_col])
        dropped_rows = original_length - len(data)
        
        if dropped_rows > 0:
            self.logger.warning(f"Dropped {dropped_rows} rows with NaN values in target column")
        
        if len(data) == 0:
            self.logger.error(f"No data remains after dropping NaN values in target column for dataset {self.dataset_name}")
            raise ValueError(f"No data remains after dropping NaN values in target column")
            
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        try:
            train_data, temp_data, train_labels, temp_labels = train_test_split(
                X, y, 
                test_size=self.split_ratios[1] + self.split_ratios[2], 
                random_state=self.random_seed, 
                stratify=y)
            
            valid_data, test_data, valid_labels, test_labels = train_test_split(
                temp_data, temp_labels, 
                test_size=self.split_ratios[2] / (self.split_ratios[1] + self.split_ratios[2]), 
                random_state=self.random_seed, 
                stratify=temp_labels)

            train_data[target_col] = train_labels
            valid_data[target_col] = valid_labels
            test_data[target_col] = test_labels
            
            self.log_info(train_data, valid_data, test_data)
            return train_data, valid_data, test_data
            
        except Exception as e:
            self.logger.error(f"Error splitting dataset {self.dataset_name}: {str(e)}")
            raise
    @abstractmethod
    def split_dataset(self, data):
        pass
    