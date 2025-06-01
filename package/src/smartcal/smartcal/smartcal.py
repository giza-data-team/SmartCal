import numpy as np
import logging
import traceback

logging.basicConfig(level=logging.ERROR)

from smartcal.meta_features_extraction.meta_features_extraction import MetaFeaturesExtractor
from smartcal.meta_model.meta_model import MetaModel
from smartcal.bayesian_optimization.calibrators_bayesian_optimization import CalibrationOptimizer
from smartcal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from smartcal.utils.cal_metrics import compute_calibration_metrics
from smartcal.utils.labels_and_probabilities import convert_one_hot_to_labels
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config_manager = ConfigurationManager()

class SmartCal:
    """
    SmartCal: A Meta-Learning-Based Framework for Probabilistic Model Calibration

    This class automates the selection and tuning of the most appropriate calibration method
    for any classification model's output probabilities. It leverages meta-features to understand
    dataset characteristics and uses a meta-learning model to make algorithm recommendations.

    Attributes:
    -----------
    metric : str
        The calibration error metric used for both recommendation and optimization. Currently supports 'ECE'.
    meta_model : MetaModel
        Meta-learning model instance trained to suggest calibrators based on extracted meta-features.
    recommended_calibrators : List[Tuple[str, float]]
        A list of calibration algorithms ranked by expected performance and confidence scores.
    fitted_calibrator : object
        Final fitted calibrator after Bayesian optimization.
    """

    def __init__(self, metric: str = 'ECE'):
        """
        Initialize the SmartCal framework with a specified calibration metric.

        Args:
            metric (str): The calibration metric to use. Only 'ECE' is supported currently.
        """
        supported_metrics = config_manager.supported_metrics
        if metric not in supported_metrics:
            raise ValueError(f"Unsupported metric '{metric}'. Supported metrics: {supported_metrics}")

        self.metric = metric
        self.meta_model = None
        self.recommended_calibrators = None
        self.fitted_calibrator = None

    def recommend_calibrators(self, y_true: np.ndarray, predictions_prob: np.ndarray, n: int = 5):
        """
        Recommend the top `n` calibration algorithms based on meta-features extracted from predictions.

        Args:
            y_true (np.ndarray): Ground truth labels (one-hot or integer encoded).
            predictions_prob (np.ndarray): Predicted class probabilities from the base model.
            n (int): Number of top calibrators to recommend (1 to 12).

        Returns:
            List[Tuple[str, float]]: Recommended calibrators and their normalized confidence scores.

        Raises:
            ValueError: For invalid inputs (NaNs, infs, shape mismatches).
        """
        if not (1 <= n <= 12):
            raise ValueError("The number of recommended calibrators `n` must be between 1 and 12.")

        # Convert one-hot to label indices if necessary
        y_true = convert_one_hot_to_labels(y_true)

        if predictions_prob.shape[0] != y_true.shape[0]:
            raise ValueError("Mismatch between number of samples in y_true and predictions_prob.")

        if np.any(np.isnan(predictions_prob)) or np.any(np.isinf(predictions_prob)):
            raise ValueError("Predicted probabilities contain NaN or inf.")

        # Extract meta-features to describe calibration complexity of task
        extractor = MetaFeaturesExtractor()
        meta_features = extractor.process_features(y_true, predictions_prob)

        # Use trained meta-model to predict top-n suitable calibration algorithms
        self.meta_model = MetaModel(metric=self.metric, top_n=n)
        recommendations = self.meta_model.predict_best_model(meta_features)

        calibrators, scores = zip(*recommendations)
        normalized_scores = np.array(scores) / sum(scores)  # Normalize confidence scores

        logging.info(f"Recommendations: {recommendations}")
        logging.info(f"Normalized Scores: {normalized_scores}")

        self.recommended_calibrators = list(zip(calibrators, normalized_scores))
        return self.recommended_calibrators

    def best_fitted_calibrator(self, y_true: np.ndarray, predictions_prob: np.ndarray, n_iter: int = 10):
        """
        Select and train the best calibration method using Bayesian hyperparameter optimization.

        The calibration metric defined in `self.metric` is minimized for selection.

        Args:
            y_true (np.ndarray): Ground truth labels (one-hot or integer format).
            predictions_prob (np.ndarray): Model output probabilities.
            n_iter (int): Total number of optimization iterations across all candidate calibrators.

        Returns:
            object: The final fitted calibrator with the best calibration score.

        Raises:
            RuntimeError: If recommendations were not generated prior to calling this method.
            ValueError: For input misalignments or invalid values.
            RuntimeError: If no calibrator could be successfully fitted.
        """
        if self.recommended_calibrators is None:
            raise RuntimeError("Call `recommend_calibrators()` before `best_fitted_calibrator()`.")

        if predictions_prob.shape[0] != y_true.shape[0]:
            raise ValueError("Mismatch between number of samples in y_true and predictions_prob.")

        if np.any(np.isnan(predictions_prob)) or np.any(np.isinf(predictions_prob)):
            raise ValueError("Predicted probabilities contain NaN or inf.")

        y_true = convert_one_hot_to_labels(y_true)
        normalized_scores = [score for _, score in self.recommended_calibrators]

        # Allocate optimization budget proportionally across recommended calibrators
        optimizer = CalibrationOptimizer(meta_model=self.meta_model)
        iteration_allocations = optimizer.allocate_iterations(normalized_scores, n_iter)

        best_calibrator = None
        best_metric_value = float('inf')

        for (calibrator_name, confidence), n_iterations in zip(self.recommended_calibrators, iteration_allocations):
            if n_iterations <= 0:
                continue

            try:
                optimization_result = optimizer.optimize_calibrator(
                    calibrator_name, predictions_prob, y_true, n_iterations, self.metric
                )

                if optimization_result is None or optimization_result.get('full_metrics') is None:
                    logging.warning(f"Optimization returned None for {calibrator_name}.")
                    continue

                current_params = optimization_result['best_params'].copy()

                try:
                    # Instantiate and fit calibrator with best found params
                    calibrator = CalibrationAlgorithmTypesEnum[calibrator_name](**current_params)
                    calibrator.fit(predictions_prob, y_true)

                    # Predict calibrated probabilities and evaluate ECE
                    cal_probs = calibrator.predict(predictions_prob)
                    cal_predicted_label = np.argmax(cal_probs, axis=1).tolist()
                    cal_metrics = compute_calibration_metrics(cal_probs, cal_predicted_label, y_true, [self.metric.lower()])
                    cal_score = cal_metrics[self.metric.lower()]

                    # Keep calibrator with lowest calibration error
                    if cal_score < best_metric_value:
                        best_metric_value = cal_score
                        best_calibrator = calibrator

                except Exception as e:
                    logging.warning(f"Failed to fit or evaluate {calibrator_name}: {str(e)}")
                    logging.debug(traceback.format_exc())
                    continue

            except Exception as e:
                logging.warning(f"Optimization failed for {calibrator_name}: {str(e)}")
                logging.debug(traceback.format_exc())
                continue

        if best_calibrator is None:
            raise RuntimeError("All calibration methods failed. Check input data and model compatibility.")

        return best_calibrator