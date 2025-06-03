import random
import numpy as np
import torch
from abc import ABC, abstractmethod

from smartcal.config.enums.dataset_types_enum import DatasetTypesEnum
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


class BaseClassifier(ABC):
    """
    Abstract base class for machine learning classifiers.
    
    Provides a standardized interface for different types of classifiers 
    and manages common functionalities like model training, predictions, and metrics.

    Supported dataset types:
        - DatasetType.TABULAR  (e.g., structured data like CSV files)
        - DatasetType.IMAGE    (e.g., image classification tasks)
        - DatasetType.LANGUAGE (e.g., text classification)
    """

    # Load default configurations
    config_manager = ConfigurationManager()
    DEFAULT_SEED = config_manager.random_seed  # Default random seed
    DEFAULT_DEVICE = config_manager.device  # Default device (e.g., "cpu" or "cuda")

    def __init__(self, model, dataset_type: DatasetTypesEnum, seed: int = DEFAULT_SEED, device: str = DEFAULT_DEVICE):
        """
        Initializes the classifier with model, dataset type, and configuration settings.

        Args:
            model: The machine learning model instance.
            dataset_type (DatasetTypesEnum): The dataset type (TABULAR, IMAGE, LANGUAGE).
            seed (int, optional): Random seed for reproducibility (default: from config).
            device (str, optional): Computation device ("cpu" or "cuda", default: from config).

        Raises:
            ValueError: If `model` is None.
            TypeError: If `dataset_type` is not an instance of DatasetType.
            TypeError: If `seed` is not an integer.
            ValueError: If `device` is not a valid PyTorch device.
        """
        if model is None:
            raise ValueError("Model cannot be None.")
        
        if not isinstance(dataset_type, DatasetTypesEnum):
            raise TypeError(f"Expected dataset_type to be DatasetType, got {type(dataset_type)}.")

        if not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, got {type(seed)}.")

        try:
            self.device = torch.device(device)
        except Exception as e:
            raise ValueError(f"Invalid device: {device}. Error: {str(e)}")

        self.seed = seed  # Stores random seed value for reproducibility
        self.model = model  # Machine learning model instance
        self.dataset_type = dataset_type  # Type of dataset being used

        # Storage for model outputs and evaluation metrics
        self.predicted_labels = []  # List of predicted labels (e.g., [0, 1, 2, 1, 0])
        self.prediction_probabilities = []  # List of probability distributions (e.g., [[0.8, 0.2], [0.6, 0.4]])
        self.ground_truth = []  # List of true labels for evaluation (e.g., [0, 1, 1, 0])

        # Metrics and performance tracking
        self.training_metrics = {}  # Stores training results (e.g., {"loss": 0.1, "accuracy": 0.95 })
        self.testing_metrics = {}  # Stores testing results (e.g., {"loss": 0.12, "accuracy": 0.92})
        self.training_history = {}  # Stores training history per epoch (e.g., {0: {"loss": 0.2, "accuracy": 0.85}})
        self.training_time = None  # Total training duration in seconds (e.g., 120.5)
        self.testing_time_predict = None  # Total testing duration (predict) in seconds (e.g., 30.2)
        self.testing_time_predictprob = None  # Total testing duration (predict_prob) in seconds (e.g., 30.2)

        # Ensure reproducibility
        self.set_random_seed(self.seed)

    def set_random_seed(self, seed: int):
        """
        Sets a fixed random seed for reproducibility.

        Args:
            seed (int): The random seed value.

        Raises:
            TypeError: If `seed` is not an integer.
        """
        if not isinstance(seed, int):
            raise TypeError(f"Seed must be an integer, got {type(seed)}.")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True     
            torch.backends.cudnn.benchmark = False

    def log_results(self):
        """
        Returns a dictionary of model-related results, including training/testing metrics.

        Returns:
            dict: A dictionary with model metadata, training/testing durations, 
                  accuracy, loss, and predictions.

        Raises:
            RuntimeError: If an error occurs while compiling results.
        """
        try:
            return {
                "model": self.model.__class__.__name__,
                "dataset_type": str(self.dataset_type),
                "device": str(self.device),
                "training_time": round(self.training_time, 5) if self.training_time is not None else None,
                "testing_time_predict": round(self.testing_time_predict, 5) if self.testing_time_predict is not None else None,
                "testing_time_predictprob": round(self.testing_time_predictprob, 5) if self.testing_time_predictprob is not None else None,
                "training_metrics": self.training_metrics or None,
                "testing_metrics": self.testing_metrics or None,
                "predicted_labels": self.predicted_labels or None,
                "prediction_probabilities": self.prediction_probabilities or None,
                "ground_truth": self.ground_truth or None,
            }
        except Exception as e:
            raise RuntimeError(f"Error while logging results: {str(e)}")

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Trains the model on the provided dataset.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predicts class labels for input data.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict_prob(self, *args, **kwargs):
        """
        Returns class probabilities instead of discrete labels.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def save_model(self, filepath):
        """
        Saves the trained model to a file.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def load_model(self, filepath):
        """
        Loads a saved model from a file.

        Must be implemented by subclasses.
        """
        pass
