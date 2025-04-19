import joblib
import pandas as pd
import numpy as np
import importlib.resources as pkg_resources
from pathlib import Path
import joblib

from smartcal.meta_model.meta_model_base import BaseMetaModel
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
import smartcal.config.resources.models as model_pkg  


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

        # Dynamically load files using stored paths
        self._model_package = model_pkg
        self.model = self._load_component(self.model_path)
        self.ordinal_encoder = self._load_component(self.ordinal_encoder_path)
        self.label_encoder = self._load_component(self.label_encoder_path)
        self.feature_selector = self._load_component(self.feature_selector_path)
        self.scaler = self._load_component(self.scaler_path)

    def _load_component(self, path):
        if path is None:
            return None

        try:
            filename = Path(path).name
            resource = pkg_resources.files(self._model_package).joinpath(filename)
            if not resource.exists():
                return None
            with pkg_resources.as_file(resource) as f:
                return joblib.load(f)
        except Exception:
            return None
    
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