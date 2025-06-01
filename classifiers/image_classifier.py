import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_lr_finder import LRFinder
import logging

from smartcal.config.enums.image_models_enum import ImageModelsEnum
from smartcal.config.enums.dataset_types_enum import DatasetTypesEnum
from classifiers.base_classifier import BaseClassifier
from smartcal.utils.cal_metrics import compute_calibration_metrics
from smartcal.utils.classification_metrics import compute_classification_metrics
from smartcal.utils.timer import time_operation
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


config_manager = ConfigurationManager()
logger = logging.getLogger(__name__)

class ImageClassifier(BaseClassifier):
    """
    ImageClassifier extends BaseClassifier to handle image classification tasks.
    It supports training, prediction, and saving/loading model states.
    """

    def __init__(self, model_enum: ImageModelsEnum, num_classes: int, *args, **kwargs):
        """
        Initialize the ImageClassifier with a specific model, number of classes, and device.

        Args:
            model_enum (ImageModelsEnum): The model architecture to use.
            num_classes (int): The number of output classes.
            *args: Variable positional arguments to pass to the model.
            **kwargs: Variable keyword arguments (can include seed and device).

        Raises:
            ValueError: If `num_classes` is not a positive integer.
            RuntimeError: If an error occurs during model initialization.
        """
        try:
            seed = kwargs.pop("seed", BaseClassifier.DEFAULT_SEED)
            device = kwargs.pop("device", BaseClassifier.DEFAULT_DEVICE)

            if not isinstance(num_classes, int) or num_classes <= 0:
                raise ValueError("num_classes must be a positive integer.")

            model = model_enum(weights=None, num_classes=num_classes)
            super().__init__(model, DatasetTypesEnum.IMAGE, seed=seed, device=device)
            self.model.to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = None
        except Exception as e:
            raise RuntimeError(f"Error initializing ImageClassifier: {e}")

    @time_operation
    def train(self, dataloader, val_loader, epochs, patience=config_manager.patience, 
              min_delta=config_manager.min_delta_early_stopper_img, lr_min=config_manager.min_lr_img,
              lr_max=config_manager.max_lr_img, num_itr=config_manager.num_itr_early_stopper_img):
        """
        Train the model using the provided dataloader with early stopping.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing the training data.
            val_loader (torch.utils.data.DataLoader): DataLoader containing the validation data.
            epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            patience (int): Number of epochs to wait for improvement before early stopping.        
        Raises:
            TypeError: If `dataloader` or `val_loader` is not a PyTorch DataLoader.
            ValueError: If `epochs`, `learning_rate`, or `patience` are not positive numbers.
            RuntimeError: If an error occurs during training.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError("dataloader must be a PyTorch DataLoader.")
        if not isinstance(val_loader, torch.utils.data.DataLoader):
            raise TypeError("val_loader must be a PyTorch DataLoader.")
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer.")

        # Check for empty dataset before starting training
        if len(dataloader.dataset) == 0:
            raise RuntimeError("Cannot train on an empty dataset.")

        try:
            # Use LR Finder to determine the optimal learning rate
            logger.info("Finding optimal learning rate using LR Finder...")
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr_min)  # Start with a small LR
            lr_finder = LRFinder(self.model, self.optimizer, self.criterion, device=self.device)
            lr_finder.range_test(dataloader, end_lr=lr_max, num_iter=num_itr)
            # Get the learning rate with the steepest gradient
            losses = lr_finder.history['loss']
            learning_rates = lr_finder.history['lr']
            
            # Find the learning rate with minimum loss
            min_loss_idx = np.argmin(losses)
            learning_rate = learning_rates[min_loss_idx]
            
            if learning_rate is None:
                learning_rate = config_manager.learning_rate
            logger.info(f"Suggested learning rate: {learning_rate}")
            lr_finder.reset()  # Reset model and optimizer
            
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            self.training_history = {}
            self.training_metrics = {}

            y_true = []
            y_pred = []
            y_pred_probabilities = []

            for epoch in range(epochs):
                # Training phase
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0

                for images, labels in dataloader:
                    if not isinstance(images, torch.Tensor):
                        raise TypeError("Each batch must contain tensors for images.")
                    if not isinstance(labels, torch.Tensor):
                        raise TypeError("Each batch must contain tensors for labels.")
                    if images.size(0) != labels.size(0):
                        raise ValueError("Mismatch between number of images and labels in a batch.")
                    if labels is None:
                        raise ValueError("Training data must include labels. Received None.")

                    if labels.dim() > 1:
                        raise ValueError(f"Expected 1D label tensor, but got shape {labels.shape}.")

                    images, labels = images.to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()

                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                # Validation phase
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        
                        val_loss += loss.item() * images.size(0)
                        _, predicted = torch.max(outputs, 1)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                        y_true.extend(labels.cpu().tolist())
                        y_pred.extend(predicted.cpu().tolist())
                        y_pred_probabilities.extend(probabilities.cpu().tolist())

                # Calculate epoch metrics
                epoch_train_loss = train_loss / train_total if train_total > 0 else float("inf")
                epoch_train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
                epoch_val_loss = val_loss / val_total if val_total > 0 else float("inf")

                # Store training history
                self.training_history[epoch] = {
                    "train_loss": epoch_train_loss,
                    "train_accuracy": epoch_train_accuracy
                }

                # Early stopping check
                if epoch_val_loss < (best_val_loss - min_delta):
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    logger.info(f"Epoch {epoch+1}: Validation loss improved to {epoch_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs. Best validation loss: {best_val_loss:.4f}")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break
                    
            # Store final metrics
            # Compute the classification metrics
            classification_metrics = compute_classification_metrics(y_true, y_pred, y_pred_probabilities)

            # Compute the calibration metrics
            calibration_metrics = compute_calibration_metrics(y_pred_probabilities, y_pred, y_true)

            self.training_metrics = {
                'loss': classification_metrics['loss'],
                'accuracy':classification_metrics['accuracy'],
                'recall_macro': classification_metrics['recall_macro'],
                'recall_micro': classification_metrics['recall_micro'],
                'recall_weighted': classification_metrics['recall_weighted'],
                'precision_macro': classification_metrics['precision_macro'],
                'precision_micro': classification_metrics['precision_micro'],
                'precision_weighted': classification_metrics['precision_weighted'],
                'f1_macro': classification_metrics['f1_macro'],
                'f1_micro': classification_metrics['f1_micro'],
                'f1_weighted': classification_metrics['f1_weighted'],

                'ece': calibration_metrics['ece'],
                'mce': calibration_metrics['mce'],
                'conf_ece': calibration_metrics['conf_ece'],
                'brier_score': calibration_metrics['brier_score'],
                'calibration_curve_mean_predicted_probs': calibration_metrics['calibration_curve_mean_predicted_probs'],
                'calibration_curve_true_probs': calibration_metrics['calibration_curve_true_probs'],
                'calibration_curve_bin_counts': calibration_metrics['calibration_curve_bin_counts'],
            }

        except ValueError as e:
            raise ValueError(f"Invalid input during training: {e}")
        except Exception as e:
            raise RuntimeError(f"Error during training: {e}")

    @time_operation
    def predict(self, dataloader):
        """
        Predict class labels using the trained model.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing test data.

        Returns:
            Numpy Array: Numpy Array of predicted class labels.

        Raises:
            TypeError: If `dataloader` is not a PyTorch DataLoader.
            RuntimeError: If the model is not trained or an error occurs during prediction.
        """
        # Ensure dataloader is valid before entering try-except
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError("dataloader must be an instance of torch.utils.data.DataLoader.")

        if not hasattr(self, "training_metrics") or not self.training_metrics:
            raise RuntimeError("The model must be trained before calling predict().")

        try:
            self.model.eval()
            all_predictions = []
            all_ground_truth = []
            total_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in dataloader:
                    if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
                        raise TypeError("Each batch must contain tensors for images and labels.")
                    if images.size(0) != labels.size(0):
                        raise ValueError("Mismatch between number of images and labels in a batch.")

                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item() * images.size(0)

                    _, predicted = torch.max(outputs, 1)
                    all_predictions.extend(predicted.cpu().tolist())
                    all_ground_truth.extend(labels.cpu().tolist())

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            self.predicted_labels = all_predictions
            self.ground_truth = all_ground_truth

            self.testing_metrics = {"loss": total_loss / total if total > 0 else float("inf"),
                                    "accuracy": 100 * correct / total if total > 0 else 0}

            return np.array(all_predictions)

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")

    @time_operation
    def predict_prob(self, dataloader):
        """
        Get prediction probabilities using the trained model.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader containing the test data.

        Returns:
            Numpy Array: Prediction probabilities for each class.

        Raises:
            TypeError: If `dataloader` is not a PyTorch DataLoader.
            RuntimeError: If the model is not trained or an error occurs during probability prediction.
        """
        try:
            if not isinstance(dataloader, torch.utils.data.DataLoader):
                raise TypeError("dataloader must be a PyTorch DataLoader.")

            if not hasattr(self, "training_metrics") or not self.training_metrics:
                raise RuntimeError("The model must be trained before calling predict_prob().")

            self.model.eval()
            all_probabilities = []

            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.to(self.device)
                    logits = self.model(images)
                    probabilities = torch.nn.functional.softmax(logits, dim=1)
                    all_probabilities.extend(probabilities.cpu().tolist())

            self.prediction_probabilities = all_probabilities
            return np.array(all_probabilities)

        except Exception as e:
            raise RuntimeError(f"Error during probability prediction: {e}")

    def save_model(self, filepath):
        """
        Save the model's state dictionary.

        Args:
            filepath (str): File path to save the model.

        Raises:
            ValueError: If `filepath` is not a string.
            RuntimeError: If an error occurs while saving the model.
        """
        try:
            if not isinstance(filepath, str):
                raise ValueError("Filepath must be a string.")
            torch.save(self.model.state_dict(), filepath)
        except Exception as e:
            raise RuntimeError(f"Error while saving model: {e}")

    def load_model(self, filepath):
        """
        Load the model's state dictionary.

        Args:
            filepath (str): File path to load the model from.

        Raises:
            ValueError: If `filepath` is not a string.
            FileNotFoundError: If the file does not exist.
            RuntimeError: If an error occurs while loading the model.
        """
        try:
            if not isinstance(filepath, str):
                raise ValueError("Filepath must be a string.")

            state_dict = torch.load(filepath, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error while loading model: {e}")
