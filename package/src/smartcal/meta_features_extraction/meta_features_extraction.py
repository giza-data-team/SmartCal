import os
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy, skew, kurtosis
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.utils.timer import time_operation
from smartcal.utils.cal_metrics import compute_calibration_metrics
from smartcal.utils.classification_metrics import compute_classification_metrics
from smartcal.utils.labels_and_probabilities import normalize_probabilities


config_manager = ConfigurationManager()

class MetaFeaturesExtractor:
    """
    A class for processing dataset features, computing model calibration metrics,
    and performing statistical analysis.
    """

    def __init__(self):
        """
        Initializes dataset attributes, model information, calibration parameters, 
        and performance metrics.
        """

    def _process_dataset_info(self, y_true, y_pred_prob):
        y_true = np.array(y_true)  # Convert to NumPy array if not already
        y_pred_prob = np.array(y_pred_prob)  # Convert to NumPy array

        # Determine number of classes
        self.num_classes = y_pred_prob.shape[1]

        # Determine number of instances
        self.num_instances = len(y_true)
        
        if(self.num_classes == 2):
            self.dataset_type = 'binary'
        else:
            self.dataset_type = 'multi-class'

    def _imbalance_ratio(self, labels):
        """
        Computes the imbalance ratio of a dataset.

        Parameters:
        - labels (list): List of labels in a classification task.

        Returns:
        - tuple: (overall imbalance ratio, per-class imbalance ratios)
        """
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        min_count = min(label_counts.values())

        overall_imbalance = round(max_count / min_count, 2) if min_count > 0 else float("inf")
        class_ratios = {label: round(max_count / count, 2) for label, count in label_counts.items()}

        return overall_imbalance, class_ratios
    
    def _get_relative_frequencies(self, data):
        """
        Computes the relative frequencies of a dataset.

        Parameters:
        - data (list or array): Input data.

        Returns:
        - np.array: Relative frequency distribution.
        """
        hist, _ = np.histogram(data, bins=config_manager.n_bin_meta_features, density=False)
        return hist / hist.sum() if hist.sum() != 0 else hist

    def _calculate_statistics(self, data, prefix):
        """
        Computes and stores statistical measures for a given dataset.

        Parameters:
        - data (array-like): Input numerical data.
        - prefix (str): Prefix for attribute names where statistics will be stored.

        Returns:
        - None (Updates instance attributes dynamically).
        """
        try:
            data = np.array(data)
            if len(data) == 0:
                return

            setattr(self, f"{prefix}_Mean", round(np.mean(data), 5))
            setattr(self, f"{prefix}_Median", round(np.median(data), 5))
            setattr(self, f"{prefix}_Std", round(np.std(data), 5))
            setattr(self, f"{prefix}_Var", round(np.var(data), 5))
            setattr(self, f"{prefix}_Entropy", round(entropy(data / np.sum(data), base=2) if np.sum(data) != 0 else 0, 5))

            # If there's only one element or variance is zero, return -999 (undefined)
            if len(data) <= 1 or round(np.var(data), 5) == 0:
                setattr(self, f"{prefix}_Skewness", -999)
                setattr(self, f"{prefix}_Kurtosis", -999)
            else:
                setattr(self, f"{prefix}_Skewness", round(skew(data), 5))
                setattr(self, f"{prefix}_Kurtosis", round(kurtosis(data), 5))

            setattr(self, f"{prefix}_Min", round(np.min(data), 5))
            setattr(self, f"{prefix}_Max", round(np.max(data), 5))
        except Exception as e:
            print(f"Error computing statistics for {prefix}: {e}")

    def _calculate_confidence_differences(self, y_true, y_pred_prob):
        """
        Computes confidence values and their differences.

        Parameters:
        - y_true (list): True labels.
        - y_pred_prob (list): Predicted probabilities.

        Returns:
        - tuple: (confidence values, absolute differences between consecutive confidences)
        """
        confidence_values = [1 - y_pred_prob[i][y_true[i]] for i in range(len(y_true))]
        differences = np.abs(np.diff(confidence_values))
        return confidence_values, differences
    
    def _get_actual_predictions(self, y_pred_prob):
        """
        Returns the predicted class labels based on the highest probability.

        Parameters:
        - y_pred_prob (list): List of predicted probability distributions.

        Returns:
        - list: Predicted class labels.
        """
        return np.argmax(y_pred_prob, axis=1)  # Returns the index of the max probability for each sample
    
    def _calculate_classification_errors(self, y_true, y_pred, y_pred_prob):
        """Computes and stores classification performance metrics including micro, macro, and weighted scores."""
        try:
            metrics = compute_classification_metrics( y_true, y_pred, y_pred_prob, zero_division=0)

            # Log loss
            setattr(self, "Classification_Log_loss", round(metrics["loss"], 5))

            # Accuracy
            setattr(self, "Classification_Accuracy", round(metrics["accuracy"], 5))

            # Precision
            setattr(self, "Classification_Precision_Micro", round(metrics["precision_micro"], 5))
            setattr(self, "Classification_Precision_Macro", round(metrics["precision_macro"], 5))
            setattr(self, "Classification_Precision_Weighted", round(metrics["precision_weighted"], 5))

            # Recall
            setattr(self, "Classification_Recall_Micro", round(metrics["recall_micro"], 5))
            setattr(self, "Classification_Recall_Macro", round(metrics["recall_macro"], 5))
            setattr(self, "Classification_Recall_Weighted", round(metrics["recall_weighted"], 5))

            # F1 Score
            setattr(self, "Classification_F1_Micro", round(metrics["f1_micro"], 5))
            setattr(self, "Classification_F1_Macro", round(metrics["f1_macro"], 5))
            setattr(self, "Classification_F1_Weighted", round(metrics["f1_weighted"], 5))

        except Exception as e:
            print(f"Error computing classification metrics: {e}")

    def _calculate_calibration_metrics(self, y_pred_prob, actual_predictions, y_true):
        metrics = compute_calibration_metrics(y_pred_prob, actual_predictions, y_true, metrics_to_compute = ['ece', 'mce', 'conf_ece', 'brier_score'])

        self.ECE_before = round(metrics["ece"], 5)
        self.MCE_before = round(metrics["mce"], 5)
        self.ConfECE_before = round(metrics["conf_ece"][0], 5)
        self.brier_score_before = round(metrics["brier_score"], 5)

    def _bhattacharyya_distance(self, p, q):
        """
        Computes the Bhattacharyya distance between two probability distributions.

        Parameters:
        - p (array-like): First probability distribution.
        - q (array-like): Second probability distribution.

        Returns:
        - float: Bhattacharyya distance.
        """
        return -np.log(np.sum(np.sqrt(p * q)))
    
    def _calculate_pairwise_distances(self, data):
        """
        Computes various distance metrics between pairs of probability distributions.
        """
        num_lists = len(data)
        epsilon = 1e-10  # Small constant to prevent division errors
        distances = {"Wasserstein": [], "KL_Divergence": [], "Jensen_Shannon": [], "Bhattacharyya": []}

        try:
            for i in range(num_lists):
                for j in range(i + 1, num_lists):
                    l1 = np.array(data[i]) / (np.sum(data[i]) + epsilon)
                    l2 = np.array(data[j]) / (np.sum(data[j]) + epsilon)

                    distances["Wasserstein"].append(round(wasserstein_distance(l1, l2), 5))
                    distances["KL_Divergence"].append(round(entropy(l1, l2 + epsilon), 5))
                    distances["Jensen_Shannon"].append(round(jensenshannon(l1, l2), 5))
                    distances["Bhattacharyya"].append(round(-np.log(np.sum(np.sqrt(l1 * l2)) + epsilon), 5))
        except Exception as e:
            print(f"Error computing pairwise distances: {e}")

        return distances

    def process_features(self, y_true, y_pred_prob):
        """
        Processes dataset features, computes statistical metrics, and model calibration details.

        Returns:
        - dict: Extracted meta-features.
        """
        # Process dataset info
        self._process_dataset_info(y_true, y_pred_prob)

        # Normalize y_pred_prob if needed
        y_pred_prob = normalize_probabilities(y_pred_prob)

        # Compute imbalance ratios
        overall_imb, class_imb = self._imbalance_ratio(y_true)
        self.class_imbalance_ratio = overall_imb

        # Compute actual predictions statistics
        actual_predictions = self._get_actual_predictions(y_pred_prob)
        actual_relative_frequencies = self._get_relative_frequencies(actual_predictions)
        self.actual_predictions_entropy = entropy(actual_relative_frequencies / np.sum(actual_relative_frequencies), base=2) if np.sum(actual_relative_frequencies) != 0 else 0
        self.actual_predictions_entropy = round(self.actual_predictions_entropy, 5)

        # Compute confidence-based statistics
        confidence_values, confidence_differences = self._calculate_confidence_differences(y_true, y_pred_prob)
        self._calculate_statistics(confidence_values, "Confidence")

        # Compute Classification errors
        self._calculate_classification_errors(y_true, actual_predictions, y_pred_prob)

        # Compute calibration metrics (before)
        self._calculate_calibration_metrics(y_pred_prob, actual_predictions, y_true)

        # Compute pairwise distances between relative frequencies
        y_pred_classes = [list(row) for row in zip(*y_pred_prob)]
        relative_frequencies = [self._get_relative_frequencies(class_) for class_ in y_pred_classes]
        pairwise_distances = self._calculate_pairwise_distances(relative_frequencies)

        for metric_name, distance_list in pairwise_distances.items():
            self._calculate_statistics(distance_list, metric_name)

        # Convert stored attributes into a dictionary
        meta_features_dict = {key: value for key, value in vars(self).items() if not callable(value) and not key.startswith("_")}

        return meta_features_dict

    def set_dataset_name(self, dataset_name):
        self.Dataset_name = dataset_name

    def set_model_name(self, model_name):
        self.Model_name = model_name
    
    def set_best_cal(self, best_cal):
        self.Best_Cal = best_cal

    def set_min_ece(self, min_ece):
        self.min_ece = min_ece

    def save_to_csv(self, csv_filename=config_manager.meta_data_file):
        """
        Saves the processed feature data to a CSV file.

        Parameters:
        - csv_filename (str): Path to the CSV file.
        """
        df = pd.DataFrame([vars(self)])

        try:
            if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0:
                existing_df = pd.read_csv(csv_filename)
                df = pd.concat([existing_df, df], ignore_index=True)
        except pd.errors.EmptyDataError:
            print(f"Warning: '{csv_filename}' is empty. Overwriting with new data.")

        df.to_csv(csv_filename, index=False)
        #print(f"CSV file '{csv_filename}' updated successfully!")