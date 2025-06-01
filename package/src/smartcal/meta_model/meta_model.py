import joblib
import pandas as pd
import numpy as np
import importlib.resources as pkg_resources
from pathlib import Path
import logging

from smartcal.meta_model.meta_model_base import BaseMetaModel
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager

try:
    import smartcal.config.resources.models as model_pkg
except ImportError:
    model_pkg = None


config_manager = ConfigurationManager()

class MetaModel(BaseMetaModel):
    """
    MetaModel loads and uses a metric-specific model and label encoder to predict the best calibration method(s)
    for a given set of input features. It supports dynamic loading of models from a package resource directory.
    """
    def __init__(self, metric: str = config_manager.metric, top_n: int = config_manager.k_recommendations):
        """
        Initialize the MetaModel with a specific metric and number of top recommendations.

        :param metric: The calibration metric to use (e.g., 'ECE', 'MCE', etc.)
        :param top_n: Number of top models to return (default from config).
        """
        super().__init__(metric=metric, top_n=top_n)
        self._model_package = model_pkg
        self.model_path = self._get_metric_model_path()
        self.label_encoder_path = self._get_metric_label_encoder_path()
        
        # Load components with validation
        self.model = self._load_component(self.model_path)
        self.label_encoder = self._load_component(self.label_encoder_path)
        
        # Validate that critical components are loaded
        self._validate_components()

    def _validate_components(self):
        """
        Validate that required components (model and label_encoder) are properly loaded.
        
        :raises RuntimeError: If critical components failed to load.
        """
        if self.model is None:
            error_msg = (
                f"Failed to load meta-model for metric '{self.metric}'. "
                f"Expected model file at: {self.model_path}. "
                f"This could be due to:\n"
                f"1. Missing model files in the package\n"
                f"2. Incorrect package installation\n"
                f"3. Missing __init__.py in models directory\n"
                f"4. Incompatible joblib version"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)
            
        if self.label_encoder is None:
            error_msg = (
                f"Failed to load label encoder for metric '{self.metric}'. "
                f"Expected encoder file at: {self.label_encoder_path}"
            )
            logging.error(error_msg)
            raise RuntimeError(error_msg)

    def _get_model_pkg_base_path(self):
        """
        Get the base filesystem path for the models package.
        :return: Path to the models package directory.
        :raises ImportError: If the models package is not available.
        """
        if self._model_package is None:
            raise ImportError("model_pkg is not available. The models package could not be imported.")
        return model_pkg.__path__[0]

    def _get_metric_model_path(self) -> str:
        """
        Construct the path to the metric-specific model file.
        :return: Path to the model file for the selected metric.
        """
        try:
            base_path = self._get_model_pkg_base_path()
            return f"{base_path}/{self.metric}/AdaBoost.joblib"
        except Exception as e:
            logging.error(f"Failed to construct model path for metric '{self.metric}': {e}")
            return None

    def _get_metric_label_encoder_path(self) -> str:
        """
        Construct the path to the metric-specific label encoder file.
        :return: Path to the label encoder file for the selected metric.
        """
        try:
            base_path = self._get_model_pkg_base_path()
            return f"{base_path}/{self.metric}/label_encoder.joblib"
        except Exception as e:
            logging.error(f"Failed to construct label encoder path for metric '{self.metric}': {e}")
            return None

    def _load_component_fallback(self, metric: str, filename: str):
        """
        Fallback method to load components using direct file system paths.
        This is used when importlib.resources approach fails.
        
        :param metric: The metric name (e.g., 'ECE', 'MCE')
        :param filename: The filename to load (e.g., 'AdaBoost.joblib', 'label_encoder.joblib')
        :return: Loaded component or None if loading fails
        """
        try:
            # Get the directory where this module is located
            current_dir = Path(__file__).parent.parent
            models_dir = current_dir / "config" / "resources" / "models" / metric
            file_path = models_dir / filename
            
            if file_path.exists():
                component = joblib.load(str(file_path))
                logging.info(f"Successfully loaded component using fallback: {file_path}")
                return component
            else:
                logging.error(f"File does not exist at fallback path: {file_path}")
                return None
                
        except Exception as e:
            logging.error(f"Fallback loading failed for {metric}/{filename}: {str(e)}")
            return None

    def _load_component(self, path):
        """
        Load a serialized component (model or encoder) from the given path using joblib.
        :param path: Path to the .joblib file.
        :return: Loaded Python object or None if not found or error.
        """
        if path is None:
            logging.warning("Component path is None, cannot load component")
            return None
            
        # First try importlib.resources approach
        if self._model_package is not None:
            try:
                filename = Path(path).name
                subdir = Path(path).parent.name
                resource = pkg_resources.files(self._model_package).joinpath(subdir).joinpath(filename)
                
                if resource.exists():
                    with pkg_resources.as_file(resource) as f:
                        component = joblib.load(f)
                        logging.info(f"Successfully loaded component: {subdir}/{filename}")
                        return component
                else:
                    logging.warning(f"Resource does not exist: {subdir}/{filename}")
                    
            except Exception as e:
                logging.warning(f"importlib.resources approach failed: {str(e)}")
        
        # Fallback to direct file system loading
        logging.info("Attempting fallback loading method...")
        filename = Path(path).name
        metric = Path(path).parent.name
        return self._load_component_fallback(metric, filename)

    def predict_best_model(self, input_features: dict) -> list:
        """
        Predict the best calibration models for the given input features.
        :param input_features: Dictionary of input features for the meta model.
        :return: List of (class_name, normalized_probability) tuples for the top recommendations.
        """
        # Validate that model is available before prediction
        if self.model is None:
            raise RuntimeError(
                f"Meta-model for metric '{self.metric}' is not available. "
                "Please check that the package was installed correctly and all model files are present."
            )
            
        try:
            X_input = pd.DataFrame([input_features])
            y_proba = self.model.predict_proba(X_input)[0]
            
            if self.label_encoder is not None:
                class_names = self.label_encoder.classes_
            elif hasattr(self.model, 'classes_'):
                class_names = self.model.classes_
            else:
                class_names = np.arange(len(y_proba))
                
            return self._select_and_normalize(y_proba, class_names)
            
        except Exception as e:
            logging.error(f"Error during model prediction: {str(e)}")
            raise RuntimeError(f"Failed to predict best model: {str(e)}")