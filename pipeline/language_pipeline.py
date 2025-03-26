import pandas as pd
import numpy as np
import logging
import re
import torch
from data_preparation.splitters.language_splitter import LanguageSplitter
from data_preparation.preprocessors.language_preprocessor import LanguagePreprocessor
from config.enums.language_models_enum import ModelType
from config.model_singleton import ModelCache
from config.configuration_manager.configuration_manager import ConfigurationManager
from config.enums.language_models_enum import LanguageModelsEnum
from pipeline.pipeline import Pipeline
from pipeline.cal_hyperparameter_tuning import tune_all_calibration
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
from utils.cal_metrics import compute_calibration_metrics

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import gc
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
        
        # Dataset splitting
        language_splitter = LanguageSplitter(dataset_name=self.config["dataset_name"],
                                            metadata_path=self.lang_metadata_path, 
                                            logs=self.config["logs"])
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
            self.ground_truth_val = [int(re.search(r'\d+', label.replace('__label__', '')).group()) for label in self.y_valid]
            self.ground_truth_test = [int(re.search(r'\d+', label.replace('__label__', '')).group()) for label in self.y_test]

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
                classifier.train(self.X_train, self.y_train, self.X_valid,self.y_valid)
                
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
                    self.true_labels_tst = self.ground_truth_test
                else:
                    self.true_labels_val = self.y_valid
                    self.true_labels_tst = self.y_test
                    
                # Calibration metrics computation
                calibration_metrics = compute_calibration_metrics(
                    probabilities=probabilities,
                    predictions=predictions,
                    true_labels=self.true_labels_tst
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
        
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Calibrating model: {model_name}")
                
                # Get probabilities
                uncalibrated_valid_probs = classifier.predict_prob(self.X_valid)
                uncalibrated_test_probs = self.results[model_name]['probabilities']
                # Normalize FastText probabilities if using FastText model
                if isinstance(model_name, LanguageModelsEnum) and model_name == LanguageModelsEnum.FASTTEXT:
                    uncalibrated_valid_probs = np.array([np.exp(probs) / np.sum(np.exp(probs)) for probs in uncalibrated_valid_probs])
                    uncalibrated_test_probs = np.array([np.exp(probs) / np.sum(np.exp(probs)) for probs in uncalibrated_test_probs])
                    
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
                    self.true_labels_tst,
                    self,
                    model_name,
                    self.config
                )
                
                # Store model results
                all_models_results[LanguageModelsEnum(model_name).name] = {
                    "train_time": {
                        "Training_time": float(self.results[model_name]['training_time']),
                        "Testing_time": float(self.results[model_name]['predict_proba_time'])
                    },
                    "train_metrics": self.results[model_name]['training_metrics'],
                    "uncalibrated_metrics_val_set": uncalibrated_valid_metrics,
                    "uncalibrated_metrics_test_set": self.results[model_name]['testing_metrics'],
                    "uncalibrated_probs_val_set": uncalibrated_valid_probs.tolist(),
                    "uncalibrated_probs_test_set": uncalibrated_test_probs.tolist(),
                    "calibration_results": calibration_results
                }
                
                # Log results
                self.logger.info(f"Model {model_name} calibration results:")
                for calibrator_name, results in calibration_results.items():
                    self.logger.info(f"{calibrator_name}:")
                    self.logger.info(f"Best hyperparameters: {results.get('best_hyperparams')}")
                    self.logger.info(f"Best ECE: {results.get('best_ece')}")
                    if 'calibrated_metrics_val_set' in results:
                        self.logger.info(f"Validation metrics: {results['calibrated_metrics_val_set']}")
                    if 'calibrated_metrics_tst_set' in results:
                        self.logger.info(f"Test metrics: {results['calibrated_metrics_tst_set']}")
                        
            except Exception as e:
                error_msg = f"Failed to calibrate model {model_name}: {str(e)}"
                self.logger.error(error_msg)
                failed_models_dict[str(model_name)] = str(error_msg)
                continue

        # Final results preparation
        dataset_info = {
            "n_instances": self.n_instances,
            "split_ratios": config_manager.split_ratios,
            "preprocessing_time": self.preprocessing_timing_info,
            "ground_truth_val_set": self.true_labels_val,
            "ground_truth_test_set": self.true_labels_tst
        }
        
        # Create final data structure
        final_data = {
            "models_results": all_models_results,
            "failed_models": failed_models_dict
        }
        
        # Save final results
        self.final_results = self.save_calibration_details(
            all_models_results=final_data,
            dataset_info=dataset_info
        )