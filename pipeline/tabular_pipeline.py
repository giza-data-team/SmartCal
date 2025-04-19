import pandas as pd
import numpy as np
import logging
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score

from data_preparation.splitters.tabular_splitter import TabularSplitter
from data_preparation.preprocessors.tabular_preprocessor import TabularPreprocessor
from smartcal.config.model_singleton import ModelCache
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from pipeline.cal_hyperparameter_tuning import tune_all_calibration
from pipeline.pipeline import Pipeline
from smartcal.utils.cal_metrics import compute_calibration_metrics


# Initialize configuration manager
config_manager = ConfigurationManager()

class TabularPipeline(Pipeline):
    """
    Pipeline implementation for tabular data calibration.
    Inherits from base Pipeline class.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize TabularPipeline with necessary attributes"""
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        # Dictionary to store classifier instances
        self.classifiers = {}
        # Dictionary to store calibration algorithms
        self.calibration_algorithms = {}
        # Dictionary to store results
        self.results = {}
        self.current_model = None
        # Track failed models
        self.failed_models = {}
        
    def load_preprocess_data(self):
        """ Load, split, and preprocess the tabular dataset """

        try:
            self.logger.info(f"Loading {self.config['task_type']} data...")
            self.tabular_metadata_path = config_manager.config_tabular

            # Load dataset from CSV
            self.dataset = pd.read_csv(self.config["dataset_path"])

            # Use provided split_seed in config or fall back to the random_seed from config
            current_split_seed = self.config.get("split_seed")  # This returns None if key doesn't exist
            if current_split_seed is None:
                current_split_seed = self.random_seed
                self.logger.info(f"No split_seed provided, using default random_seed: {current_split_seed}")
            else:
                self.logger.info(f"Using provided split_seed: {current_split_seed}")

            # Split dataset into train, validation, and test sets
            tabular_splitter = TabularSplitter(dataset_name=self.config["dataset_name"],
                                               metadata_path=self.tabular_metadata_path, 
                                               logs=self.config["logs"],
                                               random_seed=current_split_seed)
            train_tabular, valid_tabular, test_tabular = tabular_splitter.split_dataset(self.dataset)
            
            # Preprocess the data
            tabular_preprocessor = TabularPreprocessor(dataset_name=self.config["dataset_name"],
                                               metadata_path=self.tabular_metadata_path, 
                                                       logs=self.config["logs"])
            # Fit and transform training data
            self.X_train, self.y_train = tabular_preprocessor.fit_transform(train_tabular)
            # Transform validation and test data
            self.X_valid, self.y_valid = tabular_preprocessor.transform(valid_tabular)
            self.X_test, self.y_test = tabular_preprocessor.transform(test_tabular)
            # Store preprocessing timing information
            self.preprocessing_timing_info = tabular_preprocessor.get_timing()
            
            # Ensure all labels are integers
            self.y_train = self.y_train.astype(int)
            self.y_valid = self.y_valid.astype(int)
            self.y_test = self.y_test.astype(int)
            
            # Store number of instances in calibration set
            self.n_instances = valid_tabular.shape[0]
            
        except Exception as e:
            error_msg = f"Failed to load and preprocess data: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def initialize_model(self):
        """Initialize all models specified in the configuration"""
        models = list(self.config["combinations"].keys())
        for model in models:
            try:
                # Get model instance from cache using model enum
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
        """Train all initialized models and store their metrics"""
        successful_models = {}
        
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Training model: {model_name}")
                # Train the classifier
                classifier.train(X_train=self.X_train, y_train=self.y_train)
                
                # Store training results
                self.results[model_name] = {
                    'training_metrics': classifier.training_metrics,
                    'training_time': classifier.training_time
                }
                
                # Log training results
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
        
        # Check if any models were successfully trained
        if not self.classifiers:
            raise RuntimeError("All models failed to train. Pipeline cannot continue.")

    def evaluate_model(self):
        """Evaluate all trained models on test set"""
        models_to_remove = []
        
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Evaluating model: {model_name}")
                # Get predictions and probabilities
                predictions = classifier.predict(self.X_test, self.y_test)
                probabilities = classifier.predict_prob(self.X_test)
                
                # Compute calibration metrics
                calibration_metrics = compute_calibration_metrics(
                    probabilities=probabilities,
                    predictions=predictions,
                    true_labels=self.y_test
                )
                
                # Update classifier metrics
                classifier.testing_metrics.update(calibration_metrics)
                
                # Store evaluation results
                self.results[model_name].update({
                    'testing_metrics': classifier.testing_metrics,
                    'predict_proba_time': classifier.testing_time_predictprob,
                    'probabilities': probabilities
                })
                
                # Log evaluation results
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
        """Calibrate all models and evaluate their calibration performance"""
        all_models_results = {}
        failed_models_dict = {}  # New dictionary for failed models

        # Use kflods according to flag from config
        use_kfold = self.config.get("use_kfold")  # This returns None if key doesn't exist
        if use_kfold is not None:
            self.logger.info(f"Evaluating using kfold")
        
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Calibrating model: {model_name}")
                
                # Get uncalibrated probabilities for validation and test sets
                uncalibrated_valid_probs = classifier.predict_prob(self.X_valid)
                uncalibrated_test_probs = self.results[model_name]['probabilities']
                
                # Ensure probabilities are numpy arrays
                if isinstance(uncalibrated_valid_probs, list):
                    uncalibrated_valid_probs = np.array(uncalibrated_valid_probs)
                if isinstance(uncalibrated_test_probs, list):
                    uncalibrated_test_probs = np.array(uncalibrated_test_probs)
                
                # Calculate uncalibrated metrics for validation set
                valid_predictions = np.argmax(uncalibrated_valid_probs, axis=1)
                uncalibrated_valid_metrics = {
                    'loss': float(log_loss(self.y_valid, uncalibrated_valid_probs)),
                    'accuracy': float(accuracy_score(self.y_valid, valid_predictions)),
                    'precision_micro': float(precision_score(self.y_valid, valid_predictions, average='micro')),
                    'precision_macro': float(precision_score(self.y_valid, valid_predictions, average='macro')),
                    'recall_micro': float(recall_score(self.y_valid, valid_predictions, average='micro')),
                    'recall_macro': float(recall_score(self.y_valid, valid_predictions, average='macro')),
                    'f1_micro': float(f1_score(self.y_valid, valid_predictions, average='micro')),
                    'f1_macro': float(f1_score(self.y_valid, valid_predictions, average='macro'))
                }
                uncalibrated_valid_metrics.update(compute_calibration_metrics(
                    probabilities=uncalibrated_valid_probs,
                    predictions=valid_predictions,
                    true_labels=self.y_valid
                ))
                
                # Perform hyperparameter tuning for calibration
                calibration_results = tune_all_calibration(
                    uncalibrated_valid_probs,
                    uncalibrated_test_probs,
                    self.y_valid,
                    self.y_test,
                    self,
                    model_name,
                    self.config,
                    use_kfold
                )
            
                # Store comprehensive results for this model
                all_models_results[str(model_name)] = {
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
                
            except Exception as e:
                error_msg = f"Failed to calibrate model {model_name}: {str(e)}"
                self.logger.error(error_msg)
                failed_models_dict[str(model_name)] = str(error_msg) 
                continue

        # Prepare and save dataset information
        dataset_info = {
            "n_instances": self.n_instances,
            "split_ratios": config_manager.split_ratios,
            "preprocessing_time": self.preprocessing_timing_info,
            "ground_truth_val_set": self.y_valid.tolist(),
            "ground_truth_test_set": self.y_test.tolist()
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
        