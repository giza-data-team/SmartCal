from abc import ABC, abstractmethod
import numpy as np

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config_manager = ConfigurationManager()

class BaseMetaModel(ABC):
	"""
    Abstract base class for meta models that predict the best calibration methods.
    Provides interface and selection logic for top-N recommendations.
    """

	def __init__(self, metric: str = config_manager.metric, top_n: int =config_manager.k_recommendations):
		"""
        Initialize the base meta model with selection criteria.

        :param metric: The calibration metric to use (e.g., 'ECE', 'MCE', etc.)
        :param top_n: Number of top models to return (default from config).
        :raises ValueError: If top_n is not in the range 1-12.
        """
		if top_n <= 0 or top_n > 12:
			raise ValueError("top_n must be a positive integer between 1 and 12")

		self.metric = metric
		self.top_n = top_n

	def _select_and_normalize(self, probabilities: np.ndarray, class_names: np.ndarray) -> list:
		"""
        Select classes based on probabilities and criteria, then normalize probabilities.

        :param probabilities: Array of probabilities for each class.
        :param class_names: Array of class names corresponding to the probabilities.
        :return: List of (class_name, normalized_probability) tuples.
        """
		sorted_indices = np.argsort(-probabilities)
		sorted_probs = probabilities[sorted_indices]
		sorted_classes = class_names[sorted_indices]

		# Apply selection criteria
		if self.top_n is not None:
			selected_probs = sorted_probs[:self.top_n]
			selected_classes = sorted_classes[:self.top_n]
		else:
			selected_probs = sorted_probs
			selected_classes = sorted_classes

		# Normalize probabilities
		total = selected_probs.sum()
		normalized_probs = selected_probs if total == 0 else selected_probs / total

		return list(zip(selected_classes, normalized_probs))

	@abstractmethod
	def predict_best_model(self, input_features: dict) -> list:
		"""
        Predict the best models with probabilities based on input features.

        :param input_features: Dictionary of input features.
        :return: List of (class_name, probability) tuples.
        """
		pass
