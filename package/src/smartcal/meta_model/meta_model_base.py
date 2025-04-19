from abc import ABC, abstractmethod

import numpy as np


class BaseMetaModel(ABC):
	"""
    Abstract base class for meta models that predict the best calibration methods.
    """

	def __init__(self, prob_threshold: float = None, top_n: int = None):
		"""
        Initialize the meta model with selection criteria.

        :param prob_threshold: Cumulative probability threshold (0-1). Select classes until threshold is reached.
        :param top_n: Number of top models to return.
        """
		if prob_threshold is not None and not (0 <= prob_threshold <= 1):
			raise ValueError("prob_threshold must be between 0 and 1")
		if top_n is not None and top_n <= 0:
			raise ValueError("top_n must be a positive integer")

		self.prob_threshold = prob_threshold
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
		elif self.prob_threshold is not None:
			cumulative_probs = np.cumsum(sorted_probs)
			threshold_idx = np.argmax(cumulative_probs >= self.prob_threshold)
			selected_probs = sorted_probs[:threshold_idx + 1]
			selected_classes = sorted_classes[:threshold_idx + 1]
		else:
			selected_probs = sorted_probs
			selected_classes = sorted_classes

		# Normalize probabilities
		total = selected_probs.sum()
		if total == 0:
			normalized_probs = selected_probs
		else:
			normalized_probs = selected_probs / total

		return list(zip(selected_classes, normalized_probs))

	@abstractmethod
	def predict_best_model(self, input_features: dict) -> list:
		"""
        Predict the best models with probabilities based on input features.

        :param input_features: Dictionary of input features.
        :return: List of (class_name, probability) tuples.
        """
		pass
