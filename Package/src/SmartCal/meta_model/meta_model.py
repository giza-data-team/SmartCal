import joblib
import pandas as pd
import numpy as np
from SmartCal.meta_model.meta_model_base import BaseMetaModel
from SmartCal.config.configuration_manager import ConfigurationManager


class MetaModel(BaseMetaModel):
    def __init__(
            self,
            prob_threshold: float = None,
            top_n: int = None,
            model_path: str = None,
            ordinal_encoder_path: str = None,
            label_encoder_path: str = None,
            feature_selector_path: str = None,
            scaler_path: str = None
    ):
        super().__init__(prob_threshold, top_n)
        self.config_manager = ConfigurationManager()
        
        # Use provided paths or get from config
        self.model_path = model_path or self.config_manager.meta_model_path
        self.ordinal_encoder_path = ordinal_encoder_path or self.config_manager.meta_ordinal_encoder_path
        self.label_encoder_path = label_encoder_path or self.config_manager.meta_label_encoder_path
        self.feature_selector_path = feature_selector_path or self.config_manager.meta_feature_selector_path
        self.scaler_path = scaler_path or self.config_manager.meta_scaler_path

        # Load model and preprocessing components
        self.model = joblib.load(self.model_path)
        self.ordinal_encoder = self._load_component(self.ordinal_encoder_path)
        self.label_encoder = self._load_component(self.label_encoder_path)
        self.feature_selector = self._load_component(self.feature_selector_path)
        self.scaler = self._load_component(self.scaler_path)

    def _load_component(self, path):
        return joblib.load(path) if path else None

    def predict_best_model(self, input_features: dict) -> list:
        # Convert input to DataFrame
        X_input = pd.DataFrame([input_features])

        # Apply preprocessing
        if self.ordinal_encoder is not None:
            cat_columns = ['dataset_type']  # Adjust as per your model
            X_input[cat_columns] = self.ordinal_encoder.transform(X_input[cat_columns])
        if self.feature_selector is not None:
            X_input = self.feature_selector.transform(X_input)
        if self.scaler is not None:
            X_input = self.scaler.transform(X_input)

        # Get probabilities
        y_proba = self.model.predict_proba(X_input)[0]

        # Determine class names
        if self.label_encoder is not None:
            class_names = self.label_encoder.classes_
        elif hasattr(self.model, 'classes_'):
            class_names = self.model.classes_
        else:
            class_names = np.arange(len(y_proba))

        return self._select_and_normalize(y_proba, class_names)