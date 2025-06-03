import pandas as pd
import numpy as np
import logging
import re
import torch
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
import gc
import os

from data_preparation.splitters.language_splitter import LanguageSplitter
from data_preparation.preprocessors.language_preprocessor import LanguagePreprocessor
from smartcal.config.enums.language_models_enum import ModelType
from smartcal.config.model_singleton import ModelCache
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.config.enums.language_models_enum import LanguageModelsEnum
from pipeline.pipeline import Pipeline
from pipeline.cal_hyperparameter_tuning import tune_all_calibration
from smartcal.utils.cal_metrics import compute_calibration_metrics
from smartcal.config.enums.experiment_status_enum import Experiment_Status_Enum

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
config_manager = ConfigurationManager()


class LanguagePipeline(Pipeline):
    """
    Pipeline class for handling natural language processing tasks including data preparation,
    model training, evaluation, and calibration.
    """

    def __init__(self, *args, **kwargs):
        # Base class initialization and logger setup
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.classifiers = {}
        self.calibration_algorithms = {}
        self.results = {}
        # Track failed models
        self.failed_models = {}

    def load_preprocess_data(self):
        """Load and preprocess language data"""
        self.logger.info(f"Loading {self.config['task_type']} data...")
        self.lang_metadata_path = config_manager.config_language

        # Dataset loading
        self.dataset = pd.read_csv(self.config["dataset_path"])

        # Use provided split_seed in config or fall back to the random_seed from config
        current_split_seed = self.config.get("split_seed")  # This returns None if key doesn't exist
        if current_split_seed is None:
            current_split_seed = self.random_seed
            self.logger.info(f"No split_seed provided, using default random_seed: {current_split_seed}")
        else:
            self.logger.info(f"Using provided split_seed: {current_split_seed}")

        # Dataset splitting
        language_splitter = LanguageSplitter(dataset_name=self.config["dataset_name"],
                                             metadata_path=self.lang_metadata_path,
                                             logs=self.config["logs"],
                                             random_seed=current_split_seed)
        train_lang, val_lang, test_lang = language_splitter.split_dataset(self.dataset)
        self.n_instances = val_lang.shape[0]

        # Text preprocessing
        lang_preprocessor = LanguagePreprocessor(model_name=self.config["model_type"],
                                                 dataset_name=self.config["dataset_name"],
                                                 metadata_path=self.lang_metadata_path,
                                                 logs=True)

        # Data transformation
        self.X_train, self.y_train = lang_preprocessor.fit_transform(train_lang)
        self.X_valid, self.y_valid = lang_preprocessor.transform(val_lang)
        self.X_test, self.y_test = lang_preprocessor.transform(test_lang)

        # Word embedding model handling
        if self.config["model_type"] == ModelType.WordEmbeddingModel:
            self.ground_truth_val = [int(re.search(r'\d+', label.replace('__label__', '')).group()) for label in
                                     self.y_valid]
            self.ground_truth_test = [int(re.search(r'\d+', label.replace('__label__', '')).group()) for label in
                                      self.y_test]


        self.preprocessing_timing_info = lang_preprocessor.get_timing()

    def initialize_model(self):
        """Initialize language models from configuration"""
        models = list(self.config["combinations"].keys())
        for model in models:
            try:
                self.classifiers[model] = ModelCache.get_model(
                    model_enum=model,
                    task_type=self.config["task_type"],
                    seed=self.random_seed
                )
                self.logger.info(f"Successfully initialized model: {model}")
            except Exception as e:
                error_msg = f"Failed to initialize model {model}: {str(e)}"
                self.logger.error(error_msg)
                self.failed_models[model] = error_msg
                continue

        if not self.classifiers:
            raise RuntimeError("No models were successfully initialized. Pipeline cannot continue.")

    def train_model(self):
        """Train models and record metrics"""
        successful_models = {}
        torch.cuda.empty_cache()
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Training model: {model_name}")
                classifier.train(self.X_train, self.y_train, self.X_valid, self.y_valid)

                # Store results
                self.results[model_name] = {
                    'training_metrics': classifier.training_metrics,
                    'training_time': classifier.training_time
                }

                # Log results
                self.logger.info(f"Training metrics for {model_name}: {classifier.training_metrics}")
                self.logger.info(f"Training time for {model_name}: {classifier.training_time}")

                successful_models[model_name] = classifier

            except Exception as e:
                error_msg = f"Failed to train model {model_name}: {str(e)}"
                self.logger.error(error_msg)
                self.failed_models[model_name] = error_msg
                continue

        # Update classifiers dictionary to only include successful models
        self.classifiers = successful_models

        if not self.classifiers:
            raise RuntimeError("All models failed to train. Pipeline cannot continue.")

    def evaluate_model(self):
        """Evaluate models and compute metrics"""
        models_to_remove = []
        gc.collect()
        torch.cuda.empty_cache()
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Evaluating model: {model_name}")
                predictions = classifier.predict(self.X_test, self.y_test)
                probabilities = classifier.predict_prob(self.X_test)
                # Normalize FastText probabilities if using FastText model
                if isinstance(model_name, LanguageModelsEnum) and model_name == LanguageModelsEnum.FASTTEXT:
                    probabilities = np.array([np.exp(probs) / np.sum(np.exp(probs)) for probs in probabilities])

                # Label handling
                if self.config["model_type"] == ModelType.WordEmbeddingModel:
                    self.true_labels_val = self.ground_truth_val
                    self.true_labels_test = self.ground_truth_test
                else:
                    self.true_labels_val = self.y_valid
                    self.true_labels_test = self.y_test

                # Calibration metrics computation
                calibration_metrics = compute_calibration_metrics(
                    probabilities=probabilities,
                    predictions=predictions,
                    true_labels=self.true_labels_test
                )

                # Results storage
                classifier.testing_metrics.update(calibration_metrics)
                self.results[model_name].update({
                    'testing_metrics': classifier.testing_metrics,
                    'predict_proba_time': classifier.testing_time_predictprob,
                    'probabilities': probabilities
                })

                self.logger.info(f"Testing metrics for {model_name}: {classifier.testing_metrics}")

            except Exception as e:
                error_msg = f"Failed to evaluate model {model_name}: {str(e)}"
                self.logger.error(error_msg)
                self.failed_models[model_name] = error_msg
                models_to_remove.append(model_name)
                continue

        # Remove failed models
        for model_name in models_to_remove:
            self.classifiers.pop(model_name)

        if not self.classifiers:
            raise RuntimeError("All models failed evaluation. Pipeline cannot continue.")

    def calibrate_model(self):
        """Calibrate models and compute calibration metrics"""
        all_models_results = {}
        failed_models_dict = {}

        # Use kflods according to flag from config
        use_kfold = self.config.get("use_kfold")  # This returns None if key doesn't exist
        if use_kfold is not None:
            self.logger.info(f"Evaluating using kfold")

        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Calibrating model: {model_name}")

                # Get probabilities
                uncalibrated_valid_probs = classifier.predict_prob(self.X_valid)
                uncalibrated_test_probs = self.results[model_name]['probabilities']
                # Normalize FastText probabilities if using FastText model
                if isinstance(model_name, LanguageModelsEnum) and model_name == LanguageModelsEnum.FASTTEXT:
                    uncalibrated_valid_probs = np.array(
                        [np.exp(probs) / np.sum(np.exp(probs)) for probs in uncalibrated_valid_probs])
                    uncalibrated_test_probs = np.array(
                        [np.exp(probs) / np.sum(np.exp(probs)) for probs in uncalibrated_test_probs])

                # Array format conversion
                if isinstance(uncalibrated_valid_probs, list):
                    uncalibrated_valid_probs = np.array(uncalibrated_valid_probs)
                if isinstance(uncalibrated_test_probs, list):
                    uncalibrated_test_probs = np.array(uncalibrated_test_probs)

                # Compute validation metrics
                valid_predictions = np.argmax(uncalibrated_valid_probs, axis=1)

                uncalibrated_valid_metrics = {
                    'loss': float(log_loss(self.true_labels_val, uncalibrated_valid_probs)),
                    'accuracy': float(accuracy_score(self.true_labels_val, valid_predictions)),
                    # Add precision metrics
                    'precision_micro': float(precision_score(self.true_labels_val, valid_predictions, average='micro')),
                    'precision_macro': float(precision_score(self.true_labels_val, valid_predictions, average='macro')),
                    # Add recall metrics
                    'recall_micro': float(recall_score(self.true_labels_val, valid_predictions, average='micro')),
                    'recall_macro': float(recall_score(self.true_labels_val, valid_predictions, average='macro')),
                    # Add F1 metrics
                    'f1_micro': float(f1_score(self.true_labels_val, valid_predictions, average='micro')),
                    'f1_macro': float(f1_score(self.true_labels_val, valid_predictions, average='macro'))
                }

                uncalibrated_valid_metrics.update(compute_calibration_metrics(
                    probabilities=uncalibrated_valid_probs,
                    predictions=valid_predictions,
                    true_labels=self.true_labels_val
                ))

                # Calibration tuning
                calibration_results = tune_all_calibration(
                    uncalibrated_valid_probs,
                    uncalibrated_test_probs,
                    self.true_labels_val,
                    self.true_labels_test,
                    self,
                    model_name,
                    self.config,
                    use_kfold
                )
                # Prepare dataset information for this model
                dataset_info = {
                    "n_instances": self.n_instances,
                    "split_ratios": config_manager.split_ratios,
                    "preprocessing_time": self.preprocessing_timing_info,
                    "ground_truth_val_set": np.array(self.true_labels_val).tolist(),
                    "ground_truth_test_set": np.array(self.true_labels_test).tolist()
                }

                # Store model results
                model_results = {
                    "train_time": {
                        "Training_time": float(self.results[model_name]['training_time']),
                        "Testing_time": float(self.results[model_name]['predict_proba_time'])
                    },
                    "train_metrics": self.results[model_name]['training_metrics'],
                    "uncalibrated_metrics_val_set": uncalibrated_valid_metrics,
                    "uncalibrated_metrics_test_set": self.results[model_name]['testing_metrics'],
                    "uncalibrated_probs_val_set": np.array(uncalibrated_valid_probs).tolist(),
                    "uncalibrated_probs_test_set": np.array(uncalibrated_test_probs).tolist(),
                    "calibration_results": calibration_results
                }
                # Create the final structure for this model's results
                final_results = {
                    "experiment_type": self.config['experiment_type'],
                    "status": Experiment_Status_Enum.COMPLETED.value,
                    "dataset_name": self.config['dataset_name'],
                    "dataset_info": {
                        "n_instances_cal_set": dataset_info.get('n_instances', 0),
                        "split_ratios(train_cal_test)": dataset_info.get('split_ratios', []),
                        "preprocessing_time": dataset_info.get('preprocessing_time', {}),
                        "conf_ece_thresholds": self.conf_thresholds,
                        "ground_truth_val_set": np.array(self.true_labels_val).tolist(),
                        "ground_truth_test_set": np.array(self.true_labels_test).tolist()
                    },
                    "models_results": {
                        LanguageModelsEnum(model_name).name: model_results
                    },
                    "run_ids": [run_id for run_id, _, _ in self.config["combinations"].get(model_name, [])]
                }

                # Convert numpy types before saving
                final_results = self.convert_numpy_types(final_results)

                # Save results for this model immediately
                if hasattr(self, 'experiment_manager'):
                    self.experiment_manager._save_results(final_results)
                else:
                    self.logger.warning("No experiment manager available to save results")

            except Exception as e:
                error_msg = f"Failed to calibrate model {model_name}: {str(e)}"
                self.logger.error(error_msg)

                # Save failure information
                failure_info = {
                    "experiment_type": self.config['experiment_type'],
                    "status": Experiment_Status_Enum.FAILED.value,
                    "error_message": str(e),
                    "failed_model": LanguageModelsEnum(model_name).name,
                    "run_ids": [run_id for run_id, _, _ in self.config["combinations"].get(model_name, [])]
                }

                if hasattr(self, 'experiment_manager'):
                    self.experiment_manager._save_results(failure_info)

                continue

        self.logger.info("Completed calibration for all models")