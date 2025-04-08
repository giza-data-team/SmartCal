from pipeline.tabular_pipeline import TabularPipeline
from pipeline.image_pipeline import ImagePipeline
from pipeline.language_pipeline import LanguagePipeline
from Package.src.SmartCal.config.enums.dataset_types_enum  import DatasetTypesEnum
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()

class PipelineFactory:
    """
    Factory class for creating data processing pipelines.
    Implements the Factory Pattern to create appropriate pipeline instances based on data type.
    """
    
    @staticmethod
    def create_pipeline(config):
        """
        Creates and returns an appropriate pipeline instance based on the task type.
        
        Args:
            config (dict): Configuration dictionary containing task_type and other settings
                         Expected to have a 'task_type' key matching DatasetTypesEnum
        
        Returns:
            Pipeline: An instance of TabularPipeline, ImagePipeline, or LanguagePipeline
        
        Raises:
            ValueError: If the task_type specified in config is not recognized
        """
        # Create pipeline for tabular data processing
        if config["task_type"] == DatasetTypesEnum.TABULAR:
            return TabularPipeline(config)
        
        # Create pipeline for image data processing
        elif config["task_type"] == DatasetTypesEnum.IMAGE:
            return ImagePipeline(config)
        
        # Create pipeline for language/text data processing
        elif config["task_type"] == DatasetTypesEnum.LANGUAGE:
            return LanguagePipeline(config)
        
        # Raise error if task type is not supported
        else:
            raise ValueError(f"Unknown task type: {config['task_type']}")