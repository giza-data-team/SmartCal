from typing import Dict, Any
from SmartCal.config.configuration_manager import ConfigurationManager
from SmartCal.config.enums.meta_model_enum import MetaModelEnum
from .meta_model import MetaModel

class MetaModelFactory:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self._model_mapping = {
            'meta_model': MetaModel,  # Use string value directly
            # Add more model mappings here as needed
        }
        
    def create_model(self, model_type: str = None, **kwargs) -> Any:
        """
        Factory method to create meta models based on the specified type
        If no type is specified, uses the type from configuration
        """
        if model_type is None:
            model_type = self.config_manager.meta_model_type

        if model_type not in self._model_mapping:
            raise ValueError(f"Unsupported meta model type: {model_type}. Available types: {list(self._model_mapping.keys())}")

        model_class = self._model_mapping[model_type]
        return model_class(
            prob_threshold=kwargs.get('prob_threshold'),
            top_n=kwargs.get('top_n'),
            model_path=self.config_manager.meta_model_path,
            ordinal_encoder_path=self.config_manager.meta_ordinal_encoder_path,
            label_encoder_path=self.config_manager.meta_label_encoder_path,
            feature_selector_path=self.config_manager.meta_feature_selector_path,
            scaler_path=self.config_manager.meta_scaler_path
        ) 