import numpy as np
import logging
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score

from data_preparation.splitters.images_splitter import ImagesSplitter 
from data_preparation.preprocessors.images_preprocessor import ImagePreprocessor
from smartcal.config.model_singleton import ModelCache
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from pipeline.cal_hyperparameter_tuning import tune_all_calibration
from pipeline.pipeline import Pipeline
from smartcal.utils.cal_metrics import compute_calibration_metrics


config_manager = ConfigurationManager()

class ImagePipeline(Pipeline):
    """Pipeline class for handling image classification tasks"""
    def __init__(self, *args, **kwargs):
        # Base initialization
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.classifiers = {}
        self.calibration_algorithms = {}
        self.results = {}
        # Track failed models
        self.failed_models = {}

    def load_preprocess_data(self):
        """Load and preprocess image data"""
        self.logger.info(f"Loading {self.config['task_type']} data...")
        self.img_metadata_path = config_manager.config_img

        # Use provided split_seed in config or fall back to the random_seed from config
        current_split_seed = self.config.get("split_seed")  # This returns None if key doesn't exist
        if current_split_seed is None:
            current_split_seed = self.random_seed
            self.logger.info(f"No split_seed provided, using default random_seed: {current_split_seed}")
        else:
            self.logger.info(f"Using provided split_seed: {current_split_seed}")
        
        # Dataset splitting
        image_splitter = ImagesSplitter(dataset_name=self.config["dataset_name"],
                                        metadata_path=self.img_metadata_path, 
                                        logs=self.config["logs"],
                                        random_seed=current_split_seed)
        train_img, val_img, test_img = image_splitter.split_dataset()
        self.n_instances = len(val_img) # Calculate the n_instances in the val set
        
        # Split ratio calculation
        total_images = len(train_img) + len(val_img) + len(test_img)
        self.split_ratios = {
            'train_ratio': len(train_img) / total_images,
            'val_ratio': len(val_img) / total_images,
            'test_ratio': len(test_img) / total_images
        }
        
        # Image preprocessing
        preprocessor = ImagePreprocessor(dataset_name=self.config["dataset_name"],
                                       metadata_path=self.img_metadata_path, 
                                       logs=self.config["logs"])
        self.train_loader = preprocessor.fit_transform(train_img)
        self.val_loader = preprocessor.transform(val_img)
        self.test_loader = preprocessor.transform(test_img)
        self.preprocessing_timing_info = preprocessor.get_timing()

    def initialize_model(self):
        """Initialize models from configuration"""
        models = list(self.config["combinations"].keys())
        for model in models:

            try:
                self.classifiers[model] = ModelCache.get_model(
                    model_enum=model,
                    task_type=self.config["task_type"],
                    num_classes=self.config["no_classes"],
                    device=self.config["device"],
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
        
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Training model: {model_name}")
                classifier.train(dataloader=self.train_loader, 
                                 val_loader=self.val_loader,
                               epochs=config_manager.epochs)
                
                self.results[model_name] = {
                    'training_metrics': classifier.training_metrics,
                    'training_time': classifier.training_time
                }
                
                self.logger.info(f"Training metrics for {model_name}: {classifier.training_metrics}")
                self.logger.info(f"Training time for {model_name}: {classifier.training_time}")
                
                successful_models[model_name] = classifier
                
            except Exception as e:
                error_msg = f"Failed to train model {model_name}: {str(e)}"
                self.logger.error(error_msg)
                self.failed_models[model_name] = error_msg
                continue
        
        self.classifiers = successful_models
        
        if not self.classifiers:
            raise RuntimeError("All models failed to train. Pipeline cannot continue.")

    def get_labels_from_loader(self, loader):
        """Extract and convert labels from data loader"""
        labels = []
        for _, batch_labels in loader:
            labels.extend(batch_labels.numpy())
        return np.array(labels)

    def evaluate_model(self):
        """Evaluate models and compute metrics"""
        models_to_remove = []
        self.true_labels_test = self.get_labels_from_loader(self.test_loader)
        
        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Evaluating model: {model_name}")
                predictions = classifier.predict(self.test_loader)
                probabilities = classifier.predict_prob(self.test_loader)
                
                calibration_metrics = compute_calibration_metrics(
                    probabilities=probabilities,
                    predictions=predictions,
                    true_labels=self.true_labels_test
                )
                
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
        
        for model_name in models_to_remove:
            self.classifiers.pop(model_name)
        
        if not self.classifiers:
            raise RuntimeError("All models failed evaluation. Pipeline cannot continue.")

    def calibrate_model(self):
        """Calibrate models and compute calibration metrics"""
        all_models_results = {}
        failed_models_dict = {}
        val_labels = self.get_labels_from_loader(self.val_loader)

        # Use kflods according to flag from config
        use_kfold = self.config.get("use_kfold")  # This returns None if key doesn't exist
        if use_kfold is not None:
            self.logger.info(f"Evaluating using kfold")

        for model_name, classifier in self.classifiers.items():
            try:
                self.logger.info(f"Calibrating model: {model_name}")
                
                # Get probabilities
                uncalibrated_valid_probs = classifier.predict_prob(self.val_loader)
                uncalibrated_test_probs = self.results[model_name]['probabilities']
                
                # Convert to numpy arrays
                if isinstance(uncalibrated_valid_probs, list):
                    uncalibrated_valid_probs = np.array(uncalibrated_valid_probs)
                if isinstance(uncalibrated_test_probs, list):
                    uncalibrated_test_probs = np.array(uncalibrated_test_probs)
                
                # Compute validation metrics
                valid_predictions = np.argmax(uncalibrated_valid_probs, axis=1)
                
                uncalibrated_valid_metrics = {
                    'loss': float(log_loss(val_labels, uncalibrated_valid_probs)),
                    'accuracy': float(accuracy_score(val_labels, valid_predictions)),
                    # Add precision metrics
                    'precision_micro': float(precision_score(val_labels, valid_predictions, average='micro')),
                    'precision_macro': float(precision_score(val_labels, valid_predictions, average='macro')),
                    # Add recall metrics
                    'recall_micro': float(recall_score(val_labels, valid_predictions, average='micro')),
                    'recall_macro': float(recall_score(val_labels, valid_predictions, average='macro')),
                    # Add F1 metrics
                    'f1_micro': float(f1_score(val_labels, valid_predictions, average='micro')),
                    'f1_macro': float(f1_score(val_labels, valid_predictions, average='macro'))
                }
                
                uncalibrated_valid_metrics.update(compute_calibration_metrics(
                    probabilities=uncalibrated_valid_probs,
                    predictions=valid_predictions,
                    true_labels=val_labels
                ))
                
                # Perform calibration
                calibration_results = tune_all_calibration(
                    uncalibrated_valid_probs,
                    uncalibrated_test_probs,
                    val_labels,
                    self.true_labels_test,
                    self,
                    model_name,
                    self.config,
                    use_kfold
                )
                
                # Store model results
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
            "split_ratios": self.split_ratios,
            "preprocessing_time": self.preprocessing_timing_info,
            "ground_truth_val_set": val_labels.tolist(),
            "ground_truth_test_set": self.true_labels_test.tolist()
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
        