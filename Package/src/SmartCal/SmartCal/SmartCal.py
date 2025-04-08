import numpy as np
import logging
import traceback

logging.basicConfig(level=logging.ERROR)

from SmartCal.meta_features_extraction.meta_features_extraction import MetaFeaturesExtractor
from SmartCal.meta_model.meta_model import MetaModel
from bayesian_optimization.calibrators_bayesian_optimization import CalibrationOptimizer
from config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from utils.cal_metrics import compute_calibration_metrics
from utils.functions import convert_one_hot_to_labels


class SmartCal:
    """
    SmartCal: A Meta-Learning-Based Framework for Probabilistic Model Calibration

    SmartCal is an automated pipeline that selects, tunes, and applies the most suitable calibration
    method for a given classification model based on the nature of its output probabilities and ground truth labels.
    It uses meta-features to understand the dataset and recommends calibration algorithms accordingly.

    Core Features:
    --------------
    - Extracts task-specific meta-features to characterize calibration needs.
    - Uses a meta-model to recommend the top `n` suitable calibration algorithms.
    - Applies Bayesian optimization to tune each calibrator’s hyperparameters.
    - Selects the final calibrator that minimizes a specified calibration error metric.

    Calibration Metrics:
    --------------------
    - 'ECE' (Expected Calibration Error): Measures the difference between predicted confidence and actual accuracy.
      Currently, this is the only supported metric, but the architecture is extensible to support others
      (e.g., MCE, ConfECE) in future versions.

    Attributes:
    -----------
    meta_model : MetaModel
        The trained meta-learning model used for recommending calibrators.
    recommended_calibrators : List[Tuple[str, float]]
        A ranked list of calibrator names and their normalized confidence scores.
    fitted_calibrator : CalibrationAlgorithm
        The best-performing calibrator instance selected after hyperparameter tuning.

    Example:
    --------
    >>> smartcal = SmartCal()
    >>> smartcal.recommend_calibrators(y_true, predicted_probs, n=5, metric='ECE')
    >>> best_cal = smartcal.best_fitted_calibrator(y_true, predicted_probs, n_iter=10, metric='ECE')
    >>> calibrated_probs = best_cal.predict(predicted_probs)
    """

    def __init__(self):
        """
        Initialize the SmartCal class.

        This sets up the necessary internal components of the SmartCal framework:
        - `meta_model`: Will hold the trained meta-learning model once recommendations are made.
        - `recommended_calibrators`: Will contain the list of top calibration methods suggested by the meta-model.
        - `fitted_calibrator`: Will store the final calibrator object after Bayesian hyperparameter optimization.

        The object remains stateless until `recommend_calibrators()` and `best_fitted_calibrator()` are called.
        """
            
        self.meta_model = None
        self.recommended_calibrators = None
        self.fitted_calibrator = None

    def recommend_calibrators(self, y_true: np.ndarray, predictions_prob: np.ndarray, n: int = 5, metric: str = 'ECE'):
        """
        Recommend the top `n` calibration algorithms based on meta-features derived from predictions.

        This method extracts meta-features from the predicted probabilities and ground truth labels,
        and uses a trained meta-model to recommend calibration algorithms ranked by their suitability
        for the task. The ranking is based on expected performance with respect to the specified calibration metric.

        Args:
            y_true (np.ndarray): Ground truth labels, either in one-hot encoded or integer format.
            predictions_prob (np.ndarray): Model output probabilities for each class.
            n (int): Number of top calibration methods to recommend (between 1 and 12).
            metric (str): Calibration metric used for guiding the recommendation. Currently supported:
                          - 'ECE': Expected Calibration Error (default and only supported value for now).

        Returns:
            List[Tuple[str, float]]: A list of recommended calibrators with normalized confidence scores.
                                     Example: [('PLATT', 0.6), ('HISTOGRM', 0.4)]

        Raises:
            ValueError: If `n` is not within the valid range [1, 12].
            ValueError: If input arrays are misaligned or contain NaNs/Infs.
        """

        if not (1 <= n <= 12):
            raise ValueError("The number of recommended calibrators `n` must be between 1 and 12.")

        y_true = convert_one_hot_to_labels(y_true)

        if predictions_prob.shape[0] != y_true.shape[0]:
            raise ValueError("Mismatch between number of samples in y_true and predictions_prob.")

        if np.any(np.isnan(predictions_prob)) or np.any(np.isinf(predictions_prob)):
            raise ValueError("Predicted probabilities contain NaN or inf.")

        extractor = MetaFeaturesExtractor()
        meta_features = extractor.process_features(y_true, predictions_prob)

        self.meta_model = MetaModel(top_n=n)
        recommendations = self.meta_model.predict_best_model(meta_features)

        calibrators, scores = zip(*recommendations)
        normalized_scores = np.array(scores) / sum(scores)

        logging.info(f"Recommendations: {recommendations}")
        logging.info(f"Normalized Scores: {normalized_scores}")

        self.recommended_calibrators = list(zip(calibrators, normalized_scores))
        return self.recommended_calibrators

    def best_fitted_calibrator(self, y_true: np.ndarray, predictions_prob: np.ndarray, n_iter: int = 10, metric: str = 'ECE'):
        """
        Select and fit the best calibration method using Bayesian optimization.

        For each recommended calibrator, this method allocates a portion of the total optimization budget (`n_iter`)
        in proportion to the calibrator’s recommendation confidence. It tunes the calibrator's hyperparameters
        using Bayesian optimization to minimize the specified calibration error.

        Args:
            y_true (np.ndarray): Ground truth labels, either one-hot encoded or integer-encoded.
            predictions_prob (np.ndarray): Predicted class probabilities to calibrate.
            n_iter (int): Total number of Bayesian optimization iterations to distribute across calibrators.
            metric (str): Calibration metric to minimize during optimization. Currently supported:
                          - 'ECE': Expected Calibration Error (default and only supported value for now).

        Returns:
            A fitted calibrator object that yields the lowest calibration error after optimization.

        Raises:
            RuntimeError: If `recommend_calibrators()` has not been called before this method.
            ValueError: If input arrays are misaligned or contain NaNs/Infs.
            RuntimeError: If all calibration attempts fail due to data or model incompatibility.
        """

        if self.recommended_calibrators is None:
            raise RuntimeError("No recommended calibrators found. Call `recommend_calibrators()` first.")

        if predictions_prob.shape[0] != y_true.shape[0]:
            raise ValueError("Mismatch between number of samples in y_true and predictions_prob.")

        if np.any(np.isnan(predictions_prob)) or np.any(np.isinf(predictions_prob)):
            raise ValueError("Predicted probabilities contain NaN or inf.")

        y_true = convert_one_hot_to_labels(y_true)
        normalized_scores = [score for _, score in self.recommended_calibrators]

        optimizer = CalibrationOptimizer(meta_model=self.meta_model)
        iteration_allocations = optimizer.allocate_iterations(normalized_scores, n_iter)

        best_calibrator = None
        best_ece = float('inf')

        for (calibrator_name, confidence), n_iterations in zip(self.recommended_calibrators, iteration_allocations):
            if n_iterations <= 0:
                continue

            try:
                optimization_result = optimizer.optimize_calibrator(
                    calibrator_name, predictions_prob, y_true, n_iterations
                )

                if optimization_result is None or optimization_result.get('full_metrics') is None:
                    logging.warning(f"Optimization returned None for {calibrator_name}.")
                    continue

                current_params = optimization_result['best_params'].copy()
                param_str = ", ".join([f"{k}={v}" for k, v in current_params.items()]) if current_params else "None"

                try:
                    calibrator = CalibrationAlgorithmTypesEnum[calibrator_name](**current_params)
                    calibrator.fit(predictions_prob, y_true)

                    cal_probs = calibrator.predict(predictions_prob)
                    cal_predicted_label = np.argmax(cal_probs, axis=1).tolist()
                    cal_metrics = compute_calibration_metrics(cal_probs, cal_predicted_label, y_true, ['ece'])
                    cal_ece = cal_metrics['ece']

                    if cal_ece < best_ece:
                        best_ece = cal_ece
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
            raise RuntimeError("All calibration methods failed. Ensure input data is valid and compatible.")

        return best_calibrator
