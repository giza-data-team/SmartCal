import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from smartcal.config.enums.tabular_models_enum import TabularModelsEnum
from smartcal.config.enums.dataset_types_enum import DatasetTypesEnum
from classifiers.base_classifier import BaseClassifier
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.utils.timer import time_operation
from smartcal.utils.cal_metrics import compute_calibration_metrics
from smartcal.utils.classification_metrics import compute_classification_metrics


config_manager = ConfigurationManager()

class TabularClassifier(BaseClassifier):
    """
    TabularClassifier extends BaseClassifier to handle tabular data classification tasks.
    It supports training, prediction, and saving/loading model states.
    """

    def __init__(self, model_enum: TabularModelsEnum, *args, **kwargs):
        """
        Initialize the TabularClassifier with a specific model.

        Args:
            model_enum (TabularModelsEnum): The tabular model architecture to use.
            *args: Additional positional arguments for the model.
            **kwargs: Additional keyword arguments (can include seed and device).

        Raises:
            TypeError: If `model_enum` is not an instance of `TabularModelType`.
            RuntimeError: If an error occurs during model initialization.
        """
        try:
            if not isinstance(model_enum, TabularModelsEnum):
                raise TypeError(f"Expected `model_enum` to be an instance of TabularModelType, got {type(model_enum)}.")

            # Extract seed and device from kwargs or use defaults
            seed = kwargs.pop("seed", BaseClassifier.DEFAULT_SEED)
            device = kwargs.pop("device", BaseClassifier.DEFAULT_DEVICE)

            self.model = model_enum(*args, **kwargs)  # Instantiate the model
            super().__init__(self.model, DatasetTypesEnum.TABULAR, seed=seed, device=device)

        except Exception as e:
            raise RuntimeError(f"Error initializing TabularClassifier: {e}")

    @time_operation
    def train(self, X_train, y_train):
        """
        Train the model using the provided training data.

        Args:
            X_train: Features of the training dataset (array-like or DataFrame).
            y_train: Target labels of the training dataset (array-like).

        Raises:
            TypeError: If `X_train` or `y_train` is not an array-like structure.
            ValueError: If `X_train` and `y_train` have mismatched lengths.
            RuntimeError: If an error occurs during training.
        """
        try:
            if not isinstance(X_train, (pd.DataFrame, pd.Series, np.ndarray)) or not isinstance(y_train, (pd.DataFrame, pd.Series, np.ndarray)):
                raise TypeError("X_train and y_train must be either a Pandas DataFrame/Series or a NumPy array.")

            if len(X_train) != len(y_train):
                raise ValueError("X_train and y_train must have the same length.")

            self.model.fit(X_train, y_train)  # Train the model

            # Calculate training metrics
            if hasattr(self.model, "predict"):
                train_predictions = self.model.predict(X_train)

            if hasattr(self.model, "predict_proba"):
                train_probabilities = self.model.predict_proba(X_train)  # Keeps all class probabilities

            # Compute the classification metrics
            classification_metrics = compute_classification_metrics(y_train, train_predictions, train_probabilities)

            # Compute the calibration metrics
            calibration_metrics = compute_calibration_metrics(train_probabilities, train_predictions, y_train)

            self.training_metrics = {
                'loss': classification_metrics['loss'],
                'accuracy':classification_metrics['accuracy'],
                'recall_macro': classification_metrics['recall_macro'],
                'recall_micro': classification_metrics['recall_micro'],
                'recall_weighted': classification_metrics['recall_weighted'],
                'precision_macro': classification_metrics['precision_macro'],
                'precision_micro': classification_metrics['precision_micro'],
                'precision_weighted': classification_metrics['precision_weighted'],
                'f1_macro': classification_metrics['f1_macro'],
                'f1_micro': classification_metrics['f1_micro'],
                'f1_weighted': classification_metrics['f1_weighted'],

                'ece': calibration_metrics['ece'],
                'mce': calibration_metrics['mce'],
                'conf_ece': calibration_metrics['conf_ece'],
                'brier_score': calibration_metrics['brier_score'],
                'calibration_curve_mean_predicted_probs': calibration_metrics['calibration_curve_mean_predicted_probs'],
                'calibration_curve_true_probs': calibration_metrics['calibration_curve_true_probs'],
                'calibration_curve_bin_counts': calibration_metrics['calibration_curve_bin_counts'],
            }

        except Exception as e:
            raise RuntimeError(f"Error during training: {e}")

    @time_operation
    def predict(self, X, y_test=None):
        """
        Predict class labels using the trained model and calculate testing metrics.

        Args:
            X: Features of the dataset for prediction (array-like).
            y_test: (Optional) Target labels for evaluating performance (array-like).

        Returns:
            Numpy Array: Predicted class labels.

        Raises:
            TypeError: If `X` or `y_test` is not an array-like structure.
            ValueError: If `y_test` is provided but has a different length from `X`.
            RuntimeError: If an error occurs during prediction.
        """
        try:
            if not hasattr(X, "__array__"):
                raise TypeError("X must be array-like (NumPy array, DataFrame, or similar).")
            if y_test is not None:
                if not hasattr(y_test, "__array__"):
                    raise TypeError("y_test must be array-like if provided.")
                if len(X) != len(y_test):
                    raise ValueError("X and y_test must have the same length.")

            predictions = self.model.predict(X)  # Make predictions
            self.predicted_labels = predictions.tolist()

            if y_test is not None:
                self.ground_truth = y_test.tolist()
                if hasattr(self.model, "predict_proba"):
                    probabilities = self.model.predict_proba(X)
                    self.prediction_probabilities = probabilities.tolist()
                    logloss = log_loss(y_test, probabilities)
                else:
                    logloss = None

                accuracy = accuracy_score(y_test, predictions)

                self.testing_metrics = {
                    "loss": logloss,
                    "accuracy": accuracy,
                }

            return np.array(self.predicted_labels)

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")

    @time_operation
    def predict_prob(self, X):
        """
        Get prediction probabilities using the trained model.

        Args:
            X: Features of the dataset for probability prediction (array-like).

        Returns:
            Numpy Array: Prediction probabilities for each class.

        Raises:
            TypeError: If `X` is not an array-like structure.
            NotImplementedError: If the model does not support probability prediction.
            RuntimeError: If an error occurs during probability prediction.
        """
        try:
            if not hasattr(X, "__array__"):
                raise TypeError("X must be array-like (NumPy array, DataFrame, or similar).")

            # Ensure the model has a `predict_proba` method before calling it
            if not callable(getattr(self.model, "predict_proba", None)):
                raise NotImplementedError(f"{self.model.__class__.__name__} does not support probability prediction.")

            probabilities = self.model.predict_proba(X)
            # Normalize the probabilities so that they sum to 1 for each sample
            probabilities_normalized = probabilities / probabilities.sum(axis=1, keepdims=True)
            self.prediction_probabilities = probabilities_normalized.tolist()
            return np.array(self.prediction_probabilities)

        except Exception as e:
            raise RuntimeError(f"Error during probability prediction: {e}")

    def save_model(self, filepath):
        """
        Save the model's state to a file.

        Args:
            filepath (str): File path to save the model.

        Raises:
            ValueError: If `filepath` is not a string.
            RuntimeError: If an error occurs while saving the model.
        """
        try:
            if not isinstance(filepath, str):
                raise ValueError("Filepath must be a string.")
            joblib.dump(self.model, filepath)

        except Exception as e:
            raise RuntimeError(f"Error while saving model: {e}")

    def load_model(self, filepath):
        """
        Load the model's state from a file.

        Args:
            filepath (str): File path to load the model from.

        Raises:
            ValueError: If `filepath` is not a string.
            FileNotFoundError: If the file does not exist.
            RuntimeError: If an error occurs while loading the model.
        """
        try:
            if not isinstance(filepath, str):
                raise ValueError("Filepath must be a string.")
            self.model = joblib.load(filepath)

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while loading model: {e}")
