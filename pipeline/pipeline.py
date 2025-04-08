import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from Package.src.SmartCal.config.enums.calibration_metrics_enum import CalibrationMetricsEnum
from Package.src.SmartCal.config.enums.experiment_status_enum import Experiment_Status_Enum

config_manager = ConfigurationManager()

class Pipeline(ABC):
    """
    Base class for calibration pipelines.
    Provides common functionality for model calibration workflows.
    """
    
    def __init__(self, config):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Dictionary containing pipeline configuration parameters
        """
        # Store configuration and set random seeds for reproducibility
        self.config = config
        self.random_seed = config_manager.random_seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Set up logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.logger.info("Pipeline initialized...")
        
        # Initialize calibration metrics
        self.initialize_error_metrics()
        
        # Initialize pipeline status tracking
        self.pipeline_status = {
            'status': Experiment_Status_Enum.INITIALIZED.value,
            'failed_stage': None,
            'error_message': None,
            'completed_stages': []
        }

    @abstractmethod
    def load_preprocess_data(self):
        """
        Abstract method to load and preprocess data.
        """
        pass

    @abstractmethod
    def initialize_model(self):
        """
        Abstract method to initialize models.
        """
        pass

    @abstractmethod
    def train_model(self):
        """
        Abstract method to train models.
        """
        pass

    @abstractmethod
    def evaluate_model(self):
        """
        Abstract method to evaluate uncalibrated models.
        Should handle model evaluation and compute necessary metrics.
        """
        pass
    
    @abstractmethod
    def calibrate_model(self):
        """Abstract method to be implemented by concrete pipeline classes."""
        pass
    
    def initialize_error_metrics(self):
        """
        Initialize standard error metrics using the metricsenum.
        Sets up ECE, MCE, and confidence-based ECE calculators.
        """
        # Initialize Expected Calibration Error (ECE) calculator
        self.ece_calculator = CalibrationMetricsEnum.get_metric_class('ECE')(
            num_bins=config_manager.n_bins
        )
        
        # Initialize Maximum Calibration Error (MCE) calculator
        self.mce_calculator = CalibrationMetricsEnum.get_metric_class('MCE')(
            num_bins=config_manager.n_bins
        )
        
        # Get confidence thresholds from config
        self.conf_thresholds = config_manager.conf_thresholds_list
        
        # Create Confidence-based ECE calculators for different thresholds
        self.conf_ece_calculators = {
            threshold: CalibrationMetricsEnum.get_metric_class('ConfECE')(
                num_bins=config_manager.n_bins,
                confidence_threshold=threshold
            )
            for threshold in self.conf_thresholds
        }


    def save_calibration_details(self, all_models_results, dataset_info=None):
        """
        Save calibration details for all models to a single JSON file.
        """
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)

        try:
            # Convert dataset info to JSON serializable format
            dataset_info = convert_to_json_serializable(dataset_info)
            # Create the base structure with common information
            final_results = {
                "experiment_type": self.config['experiment_type'],
                "status": Experiment_Status_Enum.COMPLETED.value,
                "dataset_info": {
                    "n_instances_cal_set": dataset_info.get('n_instances', 0),
                    "split_ratios(train_cal_tst)": dataset_info.get('split_ratios', []),
                    "preprocessing_time": dataset_info.get('preprocessing_time', {}),
                    "conf_ece_thresholds": self.conf_thresholds,
                    "ground_truth_val_set": dataset_info.get('ground_truth_val_set'),
                    "ground_truth_test_set": dataset_info.get('ground_truth_test_set')
                },
                "models_results": convert_to_json_serializable(all_models_results.get("models_results", {}))}
            
            return final_results
            
        except Exception as e:
            error_msg = f"Error saving calibration details: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def save_failure_status(self, stage, error_message):
        """
        Create failure status dictionary.
        
        Args:
            stage (str): The pipeline stage where the failure occurred
            error_message (str): The error message describing the failure
            
        Returns:
            dict: Dictionary containing failure information
        """
        # Extract all run IDs from config
        run_ids = []
        if "combinations" in self.config:
            for model_name, calibrators in self.config["combinations"].items():
                # Extract just the run IDs from each calibrator configuration
                model_run_ids = [run_id for run_id, _ in calibrators]
                run_ids.extend(model_run_ids)
        failure_info = {
            "experiment_type": self.config['experiment_type'], 
            'dataset_name': self.config['dataset_name'],
            'status': Experiment_Status_Enum.FAILED.value,
            'failed_stage': stage,
            'error_message': str(error_message),
            'completed_stages': self.pipeline_status['completed_stages'],
            'timestamp': self.timestamp,
            'run_ids': run_ids  # Add just the list of run IDs
        }
        
        # Add model-specific information if available
        # if hasattr(self, 'failed_models'):
        #     failure_info['failed_models'] = self.failed_models
            
        self.logger.error(f"Pipeline failure at stage {stage}: {error_message}")
        
        return failure_info

    def execute_stage(self, stage_name, stage_function):
        """
        Execute a pipeline stage with error handling.
        
        Args:
            stage_name (str): Name of the pipeline stage
            stage_function (callable): Function to execute
            
        Returns:
            bool: True if stage completed successfully, False otherwise
        """
        try:
            self.logger.info(f"Executing stage: {stage_name}")
            stage_function()
            self.pipeline_status['completed_stages'].append(stage_name)
            return True, None
        except Exception as e:
            self.logger.error(f"Error in {stage_name}: {str(e)}")
            self.pipeline_status['status'] = Experiment_Status_Enum.FAILED.value
            self.pipeline_status['failed_stage'] = stage_name
            self.pipeline_status['error_message'] = str(e)
            failure_info = self.save_failure_status(stage_name, str(e))
            return False, failure_info

    def run(self):
        """
        Main pipeline execution method.
        Orchestrates the complete calibration workflow with error handling.
        """
        # Define pipeline stages and their corresponding methods
        pipeline_stages = [
            ('load_preprocess_data', self.load_preprocess_data),
            ('initialize_model', self.initialize_model),
            ('train_model', self.train_model),
            ('evaluate_model', self.evaluate_model),
            ('calibrate_model', self.calibrate_model)
        ]
        
        # Execute each stage
        for stage_name, stage_function in pipeline_stages:
            success, failure_info = self.execute_stage(stage_name, stage_function)
            if not success:
                # If any stage fails, return failure information dictionary
                return failure_info
        
        # If all stages complete successfully
        self.pipeline_status['status'] = Experiment_Status_Enum.COMPLETED.value
        self.logger.info("Pipeline completed successfully")
        
        # Return the final results dictionary
        return self.final_results if hasattr(self, 'final_results') else {
            'status': Experiment_Status_Enum.COMPLETED.value,
            'timestamp': self.timestamp,
            'message': 'Pipeline completed but no results were generated'
        }