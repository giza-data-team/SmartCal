# Import necessary classifiers and enums
from classifiers.tabular_classifier import TabularClassifier
from classifiers.image_classifier import ImageClassifier
from classifiers.language_classifier import LanguageClassifier
from config.enums.language_models_enum import LanguageModelsEnum

import logging
from config.enums.dataset_types_enum import DatasetTypesEnum

class ModelCache:
    """
    Singleton class that implements a caching mechanism for machine learning models.
    Ensures that models are only initialized once and reused when needed.
    """
    
    # Class-level variables for singleton pattern and model storage
    _instance = None  # Singleton instance
    _models = {}     # Dictionary to store initialized models using tuple keys
    logger = logging.getLogger(__name__)
    
    @classmethod
    def get_model(cls, model_enum, task_type, **kwargs):
        """
        Get or create a model instance based on the specified parameters.
        
        Args:
            model_enum: Enum representing the specific model type
            task_type: Type of task (TABULAR, IMAGE, or LANGUAGE)
            **kwargs: Additional arguments including:
                - seed: Random seed for reproducibility
                - num_classes: Number of classes for classification
                - device: Device to run the model on (CPU/GPU)
                
        Returns:
            Initialized model instance from cache or newly created
            
        Raises:
            ValueError: If task type is unsupported or model enum is invalid
        """
        try:
            # Create a unique key for the model using a tuple of its identifying characteristics
            model_key = (model_enum, task_type, kwargs.get('seed', None), 
                        kwargs.get('num_classes', None), kwargs.get('device', None))
            
            # Check if model exists in cache
            if model_key not in cls._models:
                cls.logger.info(f"Initializing new {task_type} model: {model_enum}")
                
                # Initialize appropriate classifier based on task type
                if task_type == DatasetTypesEnum.TABULAR:
                    # Initialize tabular classifier (e.g., Random Forest, XGBoost)
                    cls._models[model_key] = TabularClassifier(
                        model_enum=model_enum,
                        seed=kwargs.get('seed')
                    )
                elif task_type == DatasetTypesEnum.IMAGE:
                    # Initialize image classifier (e.g., MOBILENET)
                    cls._models[model_key] = ImageClassifier(
                        model_enum=model_enum,
                        num_classes=kwargs.get('num_classes'),
                        device=kwargs.get('device'),
                        seed=kwargs.get('seed')
                    )
                elif task_type == DatasetTypesEnum.LANGUAGE:
                    # Validate that model_enum is appropriate for language tasks
                    if not isinstance(model_enum, LanguageModelsEnum):
                        raise ValueError(f"Expected LanguageModelsEnum for language task, got {type(model_enum)}")
                    
                    # Initialize language classifier (e.g., BERT, GPT)
                    cls._models[model_key] = LanguageClassifier(
                        model_enum=model_enum,
                        seed=kwargs.get('seed')
                    )
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
            else:
                # Model found in cache
                cls.logger.info(f"Retrieved cached {task_type} model: {model_enum}")
            
            # Return either the cached or newly created model
            return cls._models[model_key]
                
        except Exception as e:
            # Log any errors during model initialization
            cls.logger.error(f"Error in model initialization: {e}")
            raise

    @classmethod
    def clear_cache(cls):
        """
        Clear all cached models from memory.
        Useful for freeing up memory or resetting the cache state.
        """
        cls._models.clear()
        cls.logger.info("Model cache cleared")