import logging
import json
import csv
import os
from itertools import product
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np
import ast
from datetime import datetime
from Package.src.SmartCal.config.enums.experiment_type_enum import ExperimentType
from Package.src.SmartCal.config.enums.dataset_types_enum import DatasetTypesEnum
from Package.src.SmartCal.config.enums.experiment_status_enum import Experiment_Status_Enum
from Package.src.SmartCal.config.enums.language_models_enum import ModelType
from Package.src.SmartCal.config.enums.calibration_algorithms_enum import CalibrationAlgorithmTypesEnum
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager
from pipeline.pipeline_factory import PipelineFactory
from experiment_manager.experiment_config import ExperimentConfig


class CSVExperimentManager:
    def __init__(self, dataset_type: DatasetTypesEnum, experiment_type: ExperimentType,
                 split_seed=None, use_kfold = False,
                 n_splits=ConfigurationManager.n_splits, trial_number=None ):
        """
        Initialization for running experiments and saving to CSV instead of DB

        Args:
            dataset_type: ex. Image, Tabular, Language
            experiment_type: ex. Benchmarking, KnowledgeBase
            output_dir: Directory to save CSV results
        """
        self.dataset_type = dataset_type
        self.experiment_type = experiment_type
        self.config_manager = ConfigurationManager()
        self.exp_config = ExperimentConfig()
        self.results = []


        # Initialize tracking for experiment runs
        self.completed_experiments = set()
        self.failed_experiments = set()
        self.split_seed = split_seed
        self.use_kfold = use_kfold
        self.n_splits =  n_splits
        self.trial_number = trial_number

        self.all_experiments = self._generate_experiment()

    def _generate_experiment(self):
        """
        Generates initial experiment configurations for all datasets of a given type.
        Same as original but without DB dependencies.
        """
        try:
            file_path = self.exp_config.get_config_path(self.dataset_type)
            problem_sheet = pd.read_excel(file_path)

            if problem_sheet is None:
                raise ValueError(f"No sheet found for problem type: {self.dataset_type}")

            # Filter datasets based on ExperimentType
            if self.experiment_type == ExperimentType.BENCHMARKING:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 1]
            elif self.experiment_type == ExperimentType.KNOWLEDGE_BASE:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 0]
            elif self.experiment_type == ExperimentType.BENCHMARKING_V2:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 3]
            elif self.experiment_type == ExperimentType.KNOWLEDGE_BASE_V2:
                filtered_sheet = problem_sheet[problem_sheet['Experiment Type'] == 2]

            # Convert filtered data to records
            datasets = filtered_sheet[['Dataset', 'Type', 'no. instances', 'no. classes', 'Experiment Type']].to_dict(
                'records')

            models = self.exp_config.get_model_type(self.dataset_type)
            calibration_algorithms = [
                CalibrationAlgorithmTypesEnum.BETA,
                CalibrationAlgorithmTypesEnum.TEMPERATURESCALING
            ]

            experiments_by_dataset = {}
            for dataset in datasets:
                # Create base configuration for each dataset
                base_config = {
                    "task_type": self.dataset_type,
                    "dataset_name": str(dataset['Dataset']),
                    "no_classes": int(dataset['no. classes']),
                    "no_instances": int(dataset['no. instances']),
                    "classification_type": dataset['Type'],
                    "classification_models": models,
                    "calibration_algorithms": calibration_algorithms
                }

                base_config['dataset_path'] = f"Datasets/Datasets/{self.dataset_type.name}/{dataset['Dataset']}.csv"
                experiments_by_dataset[dataset['Dataset']] = base_config

        except Exception as e:
            logging.info(f"Error in generate_experiment: {str(e)}")
            raise
        return experiments_by_dataset

    def _fetch_next(self, experiments_by_dataset: dict, add_failed: bool = False):
        """
        Modified _fetch_next to use local tracking instead of database.
        """
        experiment_by_dataset = experiments_by_dataset.copy()
        for dataset_name, dataset_config in experiment_by_dataset.items():
            # Generate all possible combinations
            all_combinations = list(product(
                dataset_config["classification_models"],
                dataset_config["calibration_algorithms"]
            ))

            # Filter available combinations
            remaining_combinations = []

            for model, cal_algo in all_combinations:
                model_cal_pair = (model.name, cal_algo.name, dataset_name)

                if model_cal_pair in self.completed_experiments:
                    continue

                if not add_failed and model_cal_pair in self.failed_experiments:
                    continue

                remaining_combinations.append((model, cal_algo))

            if not remaining_combinations:
                # If no experiments are left for this dataset, remove it
                del experiments_by_dataset[dataset_name]
                continue  # Move to next dataset

            if remaining_combinations:
                # For language tasks, group by model type
                if dataset_config["task_type"] == DatasetTypesEnum.LANGUAGE:
                    # Group models by their type
                    bert_models = {}
                    word_embedding_models = {}

                    for model, cal_algo in remaining_combinations:
                        if model.model_type == ModelType.BERT:
                            if model not in bert_models:
                                bert_models[model] = []
                            bert_models[model].append(cal_algo)
                        else:  # WordEmbeddingModel
                            if model not in word_embedding_models:
                                word_embedding_models[model] = []
                            word_embedding_models[model].append(cal_algo)

                    # Create separate configs for each model type
                    if bert_models:
                        experiments_by_dataset[dataset_name]["classification_models"] = [
                            m for m in dataset_config["classification_models"] if m not in bert_models
                        ]

                        if not experiments_by_dataset[dataset_name]["classification_models"]:
                            del experiments_by_dataset[dataset_name]
                        pipeline_config = dataset_config.copy()
                        pipeline_config.pop("classification_models")
                        pipeline_config.pop("calibration_algorithms")
                        pipeline_config["combinations"] = bert_models
                        pipeline_config["experiment_type"] = self.experiment_type
                        pipeline_config["model_type"] = ModelType.BERT
                        return pipeline_config

                    if word_embedding_models:
                        experiments_by_dataset[dataset_name]["classification_models"] = [
                            m for m in dataset_config["classification_models"] if m not in word_embedding_models
                        ]

                        if not experiments_by_dataset[dataset_name]["classification_models"]:
                            del experiments_by_dataset[dataset_name]

                        pipeline_config = dataset_config.copy()
                        pipeline_config.pop("classification_models")
                        pipeline_config.pop("calibration_algorithms")
                        pipeline_config["combinations"] = word_embedding_models
                        pipeline_config["experiment_type"] = self.experiment_type
                        pipeline_config["model_type"] = ModelType.WordEmbeddingModel

                        return pipeline_config
                else:
                    # For non-language tasks, proceed as normal
                    model_calibrations = {}
                    for model, cal_algo in remaining_combinations:
                        if model not in model_calibrations:
                            model_calibrations[model] = []
                        model_calibrations[model].append(cal_algo)

                    pipeline_config = dataset_config.copy()
                    pipeline_config.pop("classification_models")
                    pipeline_config.pop("calibration_algorithms")
                    pipeline_config["combinations"] = model_calibrations
                    pipeline_config["experiment_type"] = self.experiment_type
                    del experiments_by_dataset[dataset_name]
                    return pipeline_config

    def _run_experiment(self, config: dict, use_default_hyperparams: bool = False):
        """
        Modified to skip DB operations and generate experiment IDs locally
        """
        experiment_ids = {}
        current_id = len(self.results)

        try:
            # Iterate through models and their calibration algorithms
            for model, cal_algos in config["combinations"].items():
                model_experiments = []
                for cal_algo in cal_algos:
                    # Generate a unique ID for this experiment
                    current_id += 1

                    # Get hyperparameters for this calibration algorithm
                    cal_hyperparams = self.exp_config.get_calibration_hyperparameters(cal_algo, use_default_hyperparams)

                    # Create calibration config with algorithm-specific hyperparameters
                    cal_config = {cal_algo.value: cal_hyperparams}
                    model_experiments.append([current_id, cal_config])

                if model_experiments:
                    experiment_ids[model] = model_experiments

            # Create final pipeline configuration
            pipeline_config = {
                "task_type": config["task_type"],
                "dataset_name": config["dataset_name"],
                "dataset_path": config["dataset_path"],
                "experiment_type": config["experiment_type"],
                "combinations": experiment_ids,
                "device": self.config_manager.device,
                "logs": True,
                "no_classes": config["no_classes"],
                "split_seed": self.split_seed,
                "use_kfold": self.use_kfold,
                "n_splits": self.n_splits,
            }

            if config["task_type"] == DatasetTypesEnum.LANGUAGE:
                pipeline_config["model_type"] = config["model_type"]

            logging.info("\n Combinations are sent to the pipeline")

            # Run pipeline
            pipeline = PipelineFactory.create_pipeline(pipeline_config)
            results = pipeline.run()

            # Add dataset_name
            results["dataset_name"] = config["dataset_name"]

            # Save raw results to file for reference
            os.makedirs("pipeline_results", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            problem_type = str(config['task_type'])
            json_file_path = f"pipeline_results/{self.experiment_type.name}_{problem_type}_{timestamp}.json"
            with open(json_file_path, 'w') as json_file:
                json.dump(results, json_file, default=str, indent=4)

            # Process and save results to CSV

            self._process_results(results)

        except Exception as e:
            logging.error(f"Error running experiments: {str(e)}")
            # Mark failed experiments
            for model, cal_algos in config["combinations"].items():
                for cal_algo in cal_algos:
                    self.failed_experiments.add((model.name, cal_algo.name, config["dataset_name"]))
            raise

    def _convert_string_to_list(self, string_data):
        """
        Safely convert string representations to numeric lists
        """
        if isinstance(string_data, str):
            try:
                # Convert the string representation to a Python list
                return ast.literal_eval(string_data)
            except (ValueError, SyntaxError):
                # Handle potential errors in string format
                return []
        else:
            # If it's already a list or array, return as is
            return string_data

    def _calculate_f1_scores(self, y_true, y_probs):
        """
        Calculate F1 scores from ground truth and predicted probabilities

        Args:
            y_true: Ground truth labels
            y_probs: Predicted probabilities

        Returns:
            tuple: (macro_f1, micro_f1)
        """
        try:
            # Convert inputs to numpy arrays
            y_true = np.array(y_true)
            y_probs = np.array(y_probs)

            # Check if y_probs is 2D (needed for argmax)
            if y_probs.ndim > 1:
                predictions = y_probs.argmax(axis=1)
            else:
                # If it's already class labels, use as is
                predictions = y_probs

            # Calculate f1 scores
            macro_f1 = float(f1_score(y_true, predictions, average='macro'))
            micro_f1 = float(f1_score(y_true, predictions, average='micro'))

            return macro_f1, micro_f1
        except Exception as e:
            logging.warning(f"Error calculating F1 scores: {str(e)}")
            return None, None

    def _process_results(self, results_file: dict):
        """
        Process results and save to CSV instead of database
        """
        # experiment_type = results_file.get('experiment_type')

        try:
            # Check if this is a failed experiment at root level
            if results_file.get('status') == Experiment_Status_Enum.FAILED.value:
                experiment_ids = results_file.get('run_ids', [])
                error_message = results_file.get('error_message')
                return

            # Extract dataset-level information
            dataset_name = results_file.get('dataset_name')
            dataset_info = results_file.get('dataset_info', {})

            split_ratios = dataset_info.get('split_ratios(train_cal_tst)', [])
            train_ratio = split_ratios[0] if isinstance(split_ratios, list) and len(split_ratios) > 0 else 0
            cal_ratio = split_ratios[1] if isinstance(split_ratios, list) and len(split_ratios) > 1 else 0
            test_ratio = split_ratios[2] if isinstance(split_ratios, list) and len(split_ratios) > 2 else 0

            # Process each model's results
            model_info = results_file.get('models_results', {})
            for model_name, model_results in model_info.items():
                if 'calibration_results' in model_results:
                    # Process each calibration algorithm's results
                    for cal_algo, cal_results in model_results['calibration_results'].items():
                        run_id = cal_results.get('run_id')
                        if not run_id:
                            logging.warning(f"No run_id found for {model_name} - {cal_algo}")
                            continue

                        # Check calibration status
                        cal_status = cal_results.get('cal_status')
                        if cal_status == Experiment_Status_Enum.FAILED.value:
                            self.failed_experiments.add((model_name, cal_algo, dataset_name))
                            continue

                        # Get ground truth and predictions for test set
                        ground_truth_test = self._convert_string_to_list(dataset_info.get('ground_truth_test_set'))
                        uncal_probs_test = self._convert_string_to_list(model_results.get('uncalibrated_probs_test_set'))

                        # Calculate F1 scores for uncalibrated test set
                        uncal_test_f1_macro, uncal_test_f1_micro = None, None
                        if ground_truth_test and uncal_probs_test:
                            uncal_test_f1_macro, uncal_test_f1_micro = self._calculate_f1_scores(
                                ground_truth_test, uncal_probs_test
                            )
                        # Create a result entry
                        result_entry = {
                            "trial_no": self.trial_number,
                            "dataset_name": dataset_name,
                            "classification_model": model_name.split('.')[-1],
                            "calibration_algorithm": cal_algo,
                            "experiment_type": self.experiment_type.name,
                            "problem_type": self.dataset_type.name,
                            "timestamp": datetime.now().isoformat(),
                            "split_seed": self.split_seed,

                            # Dataset Info
                            "n_instances_cal_set": dataset_info.get('n_instances_cal_set'),
                            "split_ratio_train": train_ratio,
                            "split_ratio_cal": cal_ratio,
                            "split_ratio_test": test_ratio,
                            "ground_truth_test_set": dataset_info.get('ground_truth_test_set'),
                            "ground_truth_cal_set": dataset_info.get('ground_truth_val_set'),

                            # Timing Information
                            "preprocessing_fit_time": dataset_info.get('preprocessing_time', {}).get('fit_transform'),
                            "preprocessing_transform_time": dataset_info.get('preprocessing_time', {}).get('transform'),
                            "train_time": model_results.get('train_time', {}).get('Training_time'),
                            "test_time": model_results.get('train_time', {}).get('Testing_time'),
                            "calibration_fit_time": cal_results.get('cal_timing', {}).get('fit'),
                            "calibration_predict_time": cal_results.get('cal_timing', {}).get('predict'),

                            # Uncalibrated Metrics (train set)
                            "uncalibrated_train_loss": model_results.get('train_metrics', {}).get('loss'),
                            "uncalibrated_train_accuracy": model_results.get('train_metrics', {}).get('accuracy'),
                            "uncalibrated_train_recall_macro": model_results.get('train_metrics', {}).get(
                                'recall_macro'),
                            "uncalibrated_train_recall_micro": model_results.get('train_metrics', {}).get(
                                'recall_micro'),
                            "uncalibrated_train_recall_weighted": model_results.get('train_metrics', {}).get(
                                'recall_weighted'),
                            "uncalibrated_train_precision_macro": model_results.get('train_metrics', {}).get(
                                'precision_macro'),
                            "uncalibrated_train_precision_micro": model_results.get('train_metrics', {}).get(
                                'precision_micro'),
                            "uncalibrated_train_precision_weighted": model_results.get('train_metrics', {}).get(
                                'precision_weighted'),
                            "uncalibrated_train_f1_score_micro": model_results.get('train_metrics', {}).get('f1_micro'),
                            "uncalibrated_train_f1_score_macro": model_results.get('train_metrics', {}).get('f1_macro'),
                            "uncalibrated_train_f1_score_weighted": model_results.get('train_metrics', {}).get(
                                'f1_weighted'),
                            "uncalibrated_train_ece": model_results.get('train_metrics', {}).get('ece'),
                            "uncalibrated_train_mce": model_results.get('train_metrics', {}).get('mce'),
                            "uncalibrated_train_conf_ece": model_results.get('train_metrics', {}).get('conf_ece'),
                            "uncalibrated_train_brier_score": model_results.get('train_metrics', {}).get('brier_score'),
                            "uncalibrated_train_calibration_curve_mean_predicted_probs": model_results.get(
                                'train_metrics', {}).get('calibration_curve_mean_predicted_probs'),
                            "uncalibrated_train_calibration_curve_true_probs": model_results.get('train_metrics',
                                                                                                 {}).get(
                                'calibration_curve_true_probs'),
                            "uncalibrated_train_calibration_num_bins": model_results.get('train_metrics', {}).get(
                                'calibration_curve_bin_counts'),

                            # Uncalibrated Metrics (cal set)
                            "uncalibrated_cal_loss": model_results.get('uncalibrated_metrics_val_set', {}).get('loss'),
                            "uncalibrated_cal_accuracy": model_results.get('uncalibrated_metrics_val_set', {}).get(
                                'accuracy'),
                            "uncalibrated_cal_recall_macro": model_results.get('uncalibrated_metrics_val_set', {}).get(
                                'recall_macro'),
                            "uncalibrated_cal_recall_micro": model_results.get('uncalibrated_metrics_val_set', {}).get(
                                'recall_micro'),
                            "uncalibrated_cal_precision_macro": model_results.get('uncalibrated_metrics_val_set',
                                                                                  {}).get('precision_macro'),
                            "uncalibrated_cal_precision_micro": model_results.get('uncalibrated_metrics_val_set',
                                                                                  {}).get('precision_micro'),
                            "uncalibrated_cal_f1_score_micro": model_results.get('uncalibrated_metrics_val_set',
                                                                                 {}).get('f1_micro'),
                            "uncalibrated_cal_f1_score_macro": model_results.get('uncalibrated_metrics_val_set',
                                                                                 {}).get('f1_macro'),
                            "uncalibrated_cal_ece": model_results.get('uncalibrated_metrics_val_set', {}).get('ece'),
                            "uncalibrated_cal_mce": model_results.get('uncalibrated_metrics_val_set', {}).get('mce'),
                            "uncalibrated_cal_conf_ece": model_results.get('uncalibrated_metrics_val_set', {}).get(
                                'conf_ece'),
                            "uncalibrated_cal_brier_score": model_results.get('uncalibrated_metrics_val_set', {}).get(
                                'brier_score'),
                            "uncalibrated_cal_calibration_curve_mean_predicted_probs": model_results.get(
                                'uncalibrated_metrics_val_set', {}).get('calibration_curve_mean_predicted_probs'),
                            "uncalibrated_cal_calibration_curve_true_probs": model_results.get(
                                'uncalibrated_metrics_val_set', {}).get('calibration_curve_true_probs'),
                            "uncalibrated_cal_calibration_num_bins": model_results.get('uncalibrated_metrics_val_set',
                                                                                       {}).get(
                                'calibration_curve_bin_counts'),
                            "uncalibrated_probs_cal_set": model_results.get('uncalibrated_probs_val_set'),

                            # Uncalibrated Metrics (test)
                            "uncalibrated_test_loss": model_results.get('uncalibrated_metrics_test_set', {}).get(
                                'loss'),
                            "uncalibrated_test_accuracy": model_results.get('uncalibrated_metrics_test_set', {}).get(
                                'accuracy'),
                            "uncalibrated_test_ece": model_results.get('uncalibrated_metrics_test_set', {}).get('ece'),
                            "uncalibrated_test_mce": model_results.get('uncalibrated_metrics_test_set', {}).get('mce'),
                            "uncalibrated_test_conf_ece": model_results.get('uncalibrated_metrics_test_set', {}).get(
                                'conf_ece'),
                            "uncalibrated_test_brier_score": model_results.get('uncalibrated_metrics_test_set', {}).get(
                                'brier_score'),
                            "uncalibrated_test_calibration_curve_mean_predicted_probs": model_results.get(
                                'uncalibrated_metrics_test_set', {}).get('calibration_curve_mean_predicted_probs'),
                            "uncalibrated_test_calibration_curve_true_probs": model_results.get(
                                'uncalibrated_metrics_test_set', {}).get('calibration_curve_true_probs'),
                            "uncalibrated_test_calibration_num_bins": model_results.get('uncalibrated_metrics_test_set',
                                                                                        {}).get(
                                'calibration_curve_bin_counts'),
                            "uncalibrated_probs_test_set": model_results.get('uncalibrated_probs_test_set'),
                            "uncalibrated_test_f1_score_macro": uncal_test_f1_macro,
                            "uncalibrated_test_f1_score_micro": uncal_test_f1_micro,
                        }

                        # Add calibrated metrics if calibration was successful
                        if cal_status == Experiment_Status_Enum.COMPLETED.value:
                            # Calibrated metrics (cal set)
                            cal_probs_test = self._convert_string_to_list(cal_results.get('calibrated_probs_test_set'))
                            cal_probs_cal = self._convert_string_to_list(cal_results.get('calibrated_probs_val_set'))
                            ground_truth_cal = self._convert_string_to_list(dataset_info.get('ground_truth_val_set'))

                            # Calculate F1 scores for calibrated test set
                            cal_test_f1_macro, cal_test_f1_micro = None, None
                            if ground_truth_test and cal_probs_test:
                                cal_test_f1_macro, cal_test_f1_micro = self._calculate_f1_scores(
                                    ground_truth_test, cal_probs_test
                                )

                            # Calculate F1 scores for calibrated cal set
                            cal_cal_f1_macro, cal_cal_f1_micro = None, None
                            if ground_truth_cal and cal_probs_cal:
                                cal_cal_f1_macro, cal_cal_f1_micro = self._calculate_f1_scores(
                                    ground_truth_cal, cal_probs_cal
                                )
                            cal_val = cal_results.get('calibrated_metrics_val_set', {})
                            result_entry.update({
                                "calibrated_cal_loss": cal_val.get('loss'),
                                "calibrated_cal_accuracy": cal_val.get('accuracy'),
                                "calibrated_cal_ece": cal_val.get('ece'),
                                "calibrated_cal_mce": cal_val.get('mce'),
                                "calibrated_cal_conf_ece": cal_val.get('conf_ece'),
                                "calibrated_cal_brier_score": cal_val.get('brier_score'),
                                "calibrated_cal_calibration_curve_mean_predicted_probs": cal_val.get(
                                    'calibration_curve_mean_predicted_probs'),
                                "calibrated_cal_calibration_curve_true_probs": cal_val.get(
                                    'calibration_curve_true_probs'),
                                "calibrated_cal_calibration_num_bins": cal_val.get('calibration_curve_bin_counts'),
                                "calibrated_probs_cal_set": cal_results.get('calibrated_probs_val_set'),
                                "cal_f1_score_macro": cal_cal_f1_macro,
                                "cal_f1_score_micro": cal_cal_f1_micro,
                            })

                            # Calibrated metrics (test set)
                            cal_test = cal_results.get('calibrated_metrics_tst_set', {})
                            result_entry.update({
                                "calibrated_test_loss": cal_test.get('loss'),
                                "calibrated_test_accuracy": cal_test.get('accuracy'),
                                "calibrated_test_ece": cal_test.get('ece'),
                                "calibrated_test_mce": cal_test.get('mce'),
                                "calibrated_test_conf_ece": cal_test.get('conf_ece'),
                                "calibrated_test_brier_score": cal_test.get('brier_score'),
                                "calibrated_test_calibration_curve_mean_predicted_probs": cal_test.get(
                                    'calibration_curve_mean_predicted_probs'),
                                "calibrated_test_calibration_curve_true_probs": cal_test.get(
                                    'calibration_curve_true_probs'),
                                "calibrated_test_calibration_num_bins": cal_test.get('calibration_curve_bin_counts'),
                                "calibrated_probs_test_set": cal_results.get('calibrated_probs_test_set'),
                                "test_f1_score_macro": cal_test_f1_macro,
                                "test_f1_score_micro": cal_test_f1_micro,
                            })

                        # Add to results list
                        self.results.append(result_entry)

                        # Mark as completed
                        self.completed_experiments.add((model_name, cal_algo, dataset_name))

            # Write updated results to CSV
            self._save_to_csv()

        except Exception as e:
            logging.error(f"Error processing results: {str(e)}")
            raise

    def _save_to_csv(self):
        """
        Save current results to CSV file with fields in a specified order
        """
        if not self.results:
            logging.info("No results to save")
            return


        csv_file = os.path.join(self.config_manager.results_dir, f'dump_calibrator_results.csv')

        # Define the order of fields you want in the CSV
        ordered_fieldnames = [
            "trial_no",
            "dataset_name",
            "classification_model",
            "calibration_algorithm",
            "experiment_type",
            "problem_type",
            "timestamp",
            "split_seed",

            # Dataset Info
            "n_instances_cal_set",
            "split_ratio_train",
            "split_ratio_cal",
            "split_ratio_test",
            "ground_truth_test_set",
            "ground_truth_cal_set",

            # Timing Information
            "preprocessing_fit_time",
            "preprocessing_transform_time",
            "train_time",
            "test_time",
            "calibration_fit_time",
            "calibration_predict_time",

            # Uncalibrated Metrics (train set)
            "uncalibrated_train_loss",
            "uncalibrated_train_accuracy",
            "uncalibrated_train_recall_macro",
            "uncalibrated_train_recall_micro",
            "uncalibrated_train_recall_weighted",
            "uncalibrated_train_precision_macro",
            "uncalibrated_train_precision_micro",
            "uncalibrated_train_precision_weighted",
            "uncalibrated_train_f1_score_micro",
            "uncalibrated_train_f1_score_macro",
            "uncalibrated_train_f1_score_weighted",
            "uncalibrated_train_ece",
            "uncalibrated_train_mce",
            "uncalibrated_train_conf_ece",
            "uncalibrated_train_brier_score",
            "uncalibrated_train_calibration_curve_mean_predicted_probs",
            "uncalibrated_train_calibration_curve_true_probs",
            "uncalibrated_train_calibration_num_bins",

            # Uncalibrated Metrics (cal set)
            "uncalibrated_cal_loss",
            "uncalibrated_cal_accuracy",
            "uncalibrated_cal_recall_macro",
            "uncalibrated_cal_recall_micro",
            "uncalibrated_cal_precision_macro",
            "uncalibrated_cal_precision_micro",
            "uncalibrated_cal_f1_score_micro",
            "uncalibrated_cal_f1_score_macro",
            "uncalibrated_cal_ece",
            "uncalibrated_cal_mce",
            "uncalibrated_cal_conf_ece",
            "uncalibrated_cal_brier_score",
            "uncalibrated_cal_calibration_curve_mean_predicted_probs",
            "uncalibrated_cal_calibration_curve_true_probs",
            "uncalibrated_cal_calibration_num_bins",
            "uncalibrated_probs_cal_set",

            # Uncalibrated Metrics (test)
            "uncalibrated_test_loss",
            "uncalibrated_test_accuracy",
            "uncalibrated_test_ece",
            "uncalibrated_test_mce",
            "uncalibrated_test_conf_ece",
            "uncalibrated_test_brier_score",
            "uncalibrated_test_calibration_curve_mean_predicted_probs",
            "uncalibrated_test_calibration_curve_true_probs",
            "uncalibrated_test_calibration_num_bins",
            "uncalibrated_probs_test_set",

            # Calibrated Metrics (cal set)
            "calibrated_cal_loss",
            "calibrated_cal_accuracy",
            "calibrated_cal_ece",
            "calibrated_cal_mce",
            "calibrated_cal_conf_ece",
            "calibrated_cal_brier_score",
            "calibrated_cal_calibration_curve_mean_predicted_probs",
            "calibrated_cal_calibration_curve_true_probs",
            "calibrated_cal_calibration_num_bins",
            "calibrated_probs_cal_set",

            # Calibrated Metrics (test set)
            "calibrated_test_loss",
            "calibrated_test_accuracy",
            "calibrated_test_ece",
            "calibrated_test_mce",
            "calibrated_test_conf_ece",
            "calibrated_test_brier_score",
            "calibrated_test_calibration_curve_mean_predicted_probs",
            "calibrated_test_calibration_curve_true_probs",
            "calibrated_test_calibration_num_bins",
            "calibrated_probs_test_set",
            "cal_f1_score_macro",
            "cal_f1_score_micro",
            "test_f1_score_macro",
            "test_f1_score_micro",
            "uncalibrated_test_f1_score_macro",
            "uncalibrated_test_f1_score_micro",
        ]

        # Make sure we don't miss any fields that might be in the results
        # but not in our predefined order
        extra_fields = set()
        for result in self.results:
            extra_fields.update(set(result.keys()) - set(ordered_fieldnames))

        # Add any extra fields at the end
        final_fieldnames = ordered_fieldnames + sorted(list(extra_fields))

        # Check if file exists to determine if we need to write header
        file_exists = os.path.isfile(csv_file)

        # Open in append mode if file exists, write mode if it doesn't
        mode = 'a' if file_exists else 'w'

        with open(csv_file, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=final_fieldnames)

            # Only write header if this is a new file
            if not file_exists:
                writer.writeheader()

            writer.writerows(self.results)

        logging.info(f"Results {'appended to' if file_exists else 'saved to'} {csv_file}")

        # Clear results after saving to avoid duplicate entries on next save
        self.results = []


    def get_next_experiment(self, add_failed: bool = False, use_default_hyperparams: bool = False):
        """
        Get and run next available experiments from any dataset.
        """
        # Generate all possible combinations for all datasets
        # all_experiments = self._generate_experiment()

        next_config = self._fetch_next(self.all_experiments, add_failed)
        if next_config:
            logging.info(f"\n\nNext experiment for dataset: {next_config['dataset_name']}\n\n")
            self._run_experiment(next_config, use_default_hyperparams)
            return True
        else:
            logging.info("No new experiments available to run for any dataset")
            return False

    def run_all_experiments(self, add_failed: bool = False, use_default_hyperparams: bool = False):
        """
        Run all experiments sequentially until completion
        """
        more_experiments = True
        while more_experiments:
            more_experiments = self.get_next_experiment(add_failed, use_default_hyperparams)

        logging.info("All experiments completed")
        return self.results