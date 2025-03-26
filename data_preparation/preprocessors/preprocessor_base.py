from abc import ABC, abstractmethod
import logging
import pandas as pd
from utils.timer import time_operation
from config.configuration_manager.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()

class Preprocessor(ABC):
    def __init__(self, dataset_name, metadata_path, logs=False):
        self.dataset_name = dataset_name
        self.metadata_path = metadata_path
        self.logs = logs
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

    def log_preprocessing_info(self, stage, **kwargs):
        """Common logging method for all preprocessors"""
        if self.logs:
            self.logger.info(f"\n=== {self.dataset_name} - {stage} ===")
            for key, value in kwargs.items():
                if isinstance(value, (list, tuple)) and len(value) > 5:
                    self.logger.info(f"{key}: {value[:5]}... (showing first 5)")
                else:
                    self.logger.info(f"{key}: {value}")
            self.logger.info("=" * 50)
    
    def load_dataset_config(self):
        """
        Load and validate dataset configuration from Excel file.
        Returns a dictionary containing the configuration for the specific dataset.
        """
        try:
            df = pd.read_excel(self.metadata_path,engine="openpyxl")
            
            # Validate that dataset exists in config
            if self.dataset_name not in df['Dataset'].values:
                raise ValueError(f"Dataset '{self.dataset_name}' not found in config file")
            
            # Get the row for this dataset
            dataset_config = df[df['Dataset'] == self.dataset_name].iloc[0].to_dict()
            
            # Validate required columns based on preprocessor type
            self.validate_config_columns(dataset_config)
            
            return dataset_config
            
        except FileNotFoundError:
            self.logger.error(f"Config file {self.metadata_path} not found")
            raise
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise
        
    @abstractmethod
    def validate_config_columns(self, config):
        """
        Validate that the config has all required columns for this preprocessor type.
        To be implemented by each specific preprocessor.
        """
        pass
    
    @abstractmethod
    def fit_transform(self, data):
        pass
    
    @abstractmethod
    def transform(self, data):
        pass