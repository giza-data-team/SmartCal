import os
import re
import fasttext
import json
import torch
import numpy as np
import logging
from tempfile import TemporaryDirectory
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader

from classifiers.base_classifier import BaseClassifier
from config.enums.dataset_types_enum import DatasetTypesEnum
from config.enums.language_models_enum import LanguageModelsEnum
from utils.timer import time_operation
from config.configuration_manager import ConfigurationManager
from utils.cal_metrics import compute_calibration_metrics
from utils.classification_metrics import compute_classification_metrics


config_manager = ConfigurationManager()
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class LanguageClassifier(BaseClassifier):
    """
    A language classification model supporting both transformer-based models (e.g., MobileBERT, TinyBERT)
    and FastText. It handles training, prediction, and evaluation with customizable configurations.
    """

    def __init__(self, model_enum, *args, **kwargs):
        """
        Initialize the LanguageClassifier with the specified model type.

        Args:
            model_enum (LanguageModelType): Specifies the type of model (e.g., MOBILEBERT, TINYBERT, FASTTEXT).
            *args: Variable positional arguments for model initialization.
            **kwargs: Variable keyword arguments (can include seed, device).

        Raises:
            RuntimeError: If model initialization fails.
        """
        try:
            seed = kwargs.pop("seed", BaseClassifier.DEFAULT_SEED)
            device = kwargs.pop("device", BaseClassifier.DEFAULT_DEVICE)

            self.model_enum = model_enum
            self.num_classes = None  # Will be determined in train()
            self.model = None  # Model initialization will happen later

            if self.model_enum == LanguageModelsEnum.FASTTEXT:
                with TemporaryDirectory() as temp_dir:
                    dummy_model_path = os.path.join(temp_dir, "fasttext_dummy.txt")
                    with open(dummy_model_path, "w", encoding="utf-8") as f:
                        f.write("__label__1 Hello world\n")
                    self.model = fasttext.train_supervised(dummy_model_path, label="__label__")
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_enum.value[0], **kwargs)

            super().__init__(self.model, DatasetTypesEnum.LANGUAGE, seed=seed, device=device)

            if self.model_enum != LanguageModelsEnum.FASTTEXT:
                self.model.to(self.device)

        except Exception as e:
            raise RuntimeError(f"Error initializing LanguageClassifier: {e}")

    @time_operation
    def train(self, train_inputs, train_labels, val_inputs, val_labels,
              lr_min=None, lr_max=None, lr_finder_epochs=config_manager.language_lr_finder_epochs,
              train_epochs=config_manager.language_train_epochs,
              patience=config_manager.language_patience, min_delta=config_manager.language_min_delta,
              monitor_metric=config_manager.language_monitor_metric,
              batch_size=config_manager.language_batch_size, training_args_dict=None):
        """
        Train the model using preprocessed inputs with LR finder and early stopping.

        Args:
            train_inputs: Tokenized input text.
            train_labels: Integer labels for classification (training set).
            val_inputs: Tokenized input text.
            val_labels: Integer labels for classification (validation set).
            training_args_dict (dict, optional): Additional arguments for training.
            lr_min (float): Minimum learning rate for LR Finder.
            lr_max (float): Maximum learning rate for LR Finder.
            lr_finder_epochs (int): Epochs for LR Finder.
            train_epochs (int): Total epochs for actual training.
            patience (int): Early stopping patience (stops training if no improvement).
            batch_size (int): Batch size for training and evaluation.
            monitor_metric (str): Metric to monitor for early stopping ('loss' or 'accuracy').
            min_delta (float): Minimum required improvement to reset early stopping patience.
        """
        unique_labels = sorted(set(train_labels))
        self.num_classes = len(unique_labels)

        if self.model_enum == LanguageModelsEnum.FASTTEXT:
            if (lr_min == None):
                lr_min = config_manager.language_fasttext_lr_min
            if (lr_max == None):
                lr_max = config_manager.language_fasttext_lr_max

            train_labels = [f"__label__{label}" if not str(label).startswith("__label__") else label for label in
                            train_labels]
            val_labels = [f"__label__{label}" if not str(label).startswith("__label__") else label for label in
                          val_labels] if val_labels else None

            self._train_fasttext(train_inputs, train_labels, val_inputs, val_labels, unique_labels,
                                 lr_min, lr_max, lr_finder_epochs, train_epochs, patience,
                                 min_delta, monitor_metric)
        else:
            if (lr_min == None):
                lr_min = 1e-5
            if (lr_max == None):
                lr_max = 5e-3

            # Initialize Transformer model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_enum.value[0], num_labels=self.num_classes
            )
            self.model.to(self.device)

            # Call super().__init__() now that self.model is defined
            super().__init__(self.model, self.dataset_type, seed=self.seed, device=self.device)

            self._train_transformer(train_inputs, train_labels, val_inputs, val_labels, training_args_dict,
                                    lr_min, lr_max, lr_finder_epochs, train_epochs, patience,
                                    batch_size, monitor_metric, min_delta)

    def _train_transformer(self, train_inputs, train_labels, val_inputs, val_labels, training_args_dict,
                           lr_min, lr_max, lr_finder_epochs, train_epochs, patience,
                           batch_size, monitor_metric, min_delta):
        """
        Train a transformer-based model with:
        - Learning Rate Finder
        - Early Stopping with Best Model Rollback
        - Optimized Training Loop with Accuracy Tracking

        Args:
            train_inputs: Tokenized input text (`input_ids`, `attention_mask`).
            train_labels: Integer labels for classification (training set).
            val_inputs: Tokenized input text (`input_ids`, `attention_mask`).
            val_labels: Integer labels for classification (validation set).
            training_args_dict (dict, optional): Additional arguments for training.
            lr_min (float): Minimum learning rate for LR Finder.
            lr_max (float): Maximum learning rate for LR Finder.
            lr_finder_epochs (int): Epochs for LR Finder.
            train_epochs (int): Total epochs for actual training.
            patience (int): Early stopping patience (stops training if no improvement).
            batch_size (int): Batch size for training and evaluation.
            monitor_metric (str): Metric to monitor for early stopping ('loss' or 'accuracy').
            min_delta (float): Minimum required improvement to reset early stopping patience.
        """

        # Validate input arguments
        if not isinstance(patience, int) or patience < 1:
            raise ValueError("`patience` must be an integer greater than or equal to 1.")

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("`batch_size` must be an integer greater than or equal to 1.")

        if monitor_metric not in ["loss", "accuracy"]:
            raise ValueError("`monitor_metric` must be either 'loss' or 'accuracy'.")

        # Convert datasets to PyTorch format
        train_data = Dataset.from_dict({
            "input_ids": train_inputs["input_ids"],
            "attention_mask": train_inputs["attention_mask"],
            "labels": train_labels
        })
        train_data.set_format(type="torch")

        val_data = Dataset.from_dict({
            "input_ids": val_inputs["input_ids"],
            "attention_mask": val_inputs["attention_mask"],
            "labels": val_labels
        })
        val_data.set_format(type="torch")

        with TemporaryDirectory() as temp_dir:
            logging.info(f"Running LR Finder: Range {lr_min:.1e} → {lr_max:.1e}")

            # Step 1: Learning Rate Finder
            def find_best_lr():
                trainer = Trainer(
                    model=self.model,
                    args=TrainingArguments(
                        output_dir=temp_dir,
                        eval_strategy="epoch",
                        per_device_train_batch_size=batch_size,
                        num_train_epochs=lr_finder_epochs,
                        report_to="none"
                    ),
                    train_dataset=train_data,
                    eval_dataset=val_data
                )

                optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr_min)
                dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

                num_steps = len(dataloader) * lr_finder_epochs
                if num_steps == 0:
                    raise RuntimeError("No steps found in dataloader. Check dataset.")

                lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), num=num_steps)
                losses = []

                self.model.train()
                for step, batch in enumerate(dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    optimizer.param_groups[0]["lr"] = lrs[step]
                    optimizer.zero_grad()

                    try:
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                    except RuntimeError as e:
                        raise RuntimeError(f"RuntimeError during LR Finder: {e}")

                    losses.append(loss.item())

                    if len(losses) > 5 and losses[-1] > losses[-2] * 1.5:
                        logging.warning("Loss is diverging! Stopping LR Finder.")
                        break

                if len(losses) < 3 or np.std(losses) < 1e-3:
                    logging.warning("Loss is too flat, using fallback LR")
                    return (lr_max + lr_min) / 2

                smoothed_loss = np.convolve(losses, np.ones(5) / 5, mode="valid")
                best_lr = lrs[np.argmin(smoothed_loss)]
                logging.info(f"Best LR Found: {best_lr:.6f}")
                return best_lr

            best_lr = find_best_lr()

            # Step 2: Train with Early Stopping and Best Model Rollback
            training_args = TrainingArguments(
                output_dir=temp_dir,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model=monitor_metric,
                save_total_limit=3,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=train_epochs,
                logging_dir=f"{temp_dir}/logs",
                report_to="none",
                learning_rate=best_lr
            )

            if training_args_dict:
                for key, value in training_args_dict.items():
                    setattr(training_args, key, value)

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data,
                compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))},
                callbacks=[EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=min_delta)]
            )

            train_result = trainer.train()

            # Compute training accuracy manually
            train_predictions = trainer.predict(train_data).predictions.argmax(axis=-1)
            train_accuracy = accuracy_score(train_labels, train_predictions) if train_labels is not None else None
            train_loss = train_result.metrics.get("train_loss", None)

            # Explicitly Reload Best Checkpoint Before Final Evaluation
            best_checkpoint = trainer.state.best_model_checkpoint
            if best_checkpoint:
                logging.info(f"Loading best model from {best_checkpoint}")
                if os.path.exists(best_checkpoint):
                    self.model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
                else:
                    logging.warning("Best checkpoint directory not found! Keeping the last trained model.")

            # Step 4: Recalculate Validation Accuracy
            eval_results = trainer.evaluate()
            val_loss = eval_results.get("eval_loss", None)

            # Manually Compute Validation Accuracy
            val_predictions = trainer.predict(val_data).predictions.argmax(axis=-1)
            val_accuracy = accuracy_score(val_labels, val_predictions) if val_labels is not None else None

            # Predict on Validation Set
            val_preds = trainer.predict(val_data)
            y_pred_prob = val_preds.predictions  # Raw logits or probabilities
            y_pred = y_pred_prob.argmax(axis=-1)  # Convert logits to predicted class
            y_true = val_labels  # Ground truth labels

            logging.info(f"Final Training Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2%}")
            logging.info(f"Final Validation Loss (after rollback): {val_loss:.4f}, Accuracy: {val_accuracy:.2%}")

            # Store final metrics
            # Compute the classification metrics
            classification_metrics = compute_classification_metrics(y_true, y_pred, y_pred_prob)

            # Compute the calibration metrics
            calibration_metrics = compute_calibration_metrics(y_pred_prob, y_pred, y_true)

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

    def _train_fasttext(self, train_texts, train_labels, val_texts, val_labels, unique_labels,
                        lr_min, lr_max, lr_finder_epochs, train_epochs, patience,
                        min_delta, monitor_metric):
        """
        Train a fasttext model with:
        - Learning Rate Finder
        - Early Stopping with Best Model Rollback
        - Optimized Training Loop with Accuracy Tracking

        Args:
            train_inputs: Tokenized input text.
            train_labels: Integer labels for classification (training set).
            val_inputs: Tokenized input text.
            val_labels: Integer labels for classification (validation set).
            lr_min (float): Minimum learning rate for LR Finder.
            lr_max (float): Maximum learning rate for LR Finder.
            lr_finder_epochs (int): Epochs for LR Finder.
            train_epochs (int): Total epochs for actual training.
            patience (int): Early stopping patience (stops training if no improvement).
            monitor_metric (str): Metric to monitor for early stopping ('loss' or 'accuracy').
            min_delta (float): Minimum required improvement to reset early stopping patience.
        """
        if monitor_metric not in ["loss", "accuracy"]:
            raise ValueError("`monitor_metric` must be either 'loss' or 'accuracy'.")

        # Create a label-to-index mapping
        label_map = {label: idx for idx, label in enumerate(unique_labels)}

        # Convert labels to numeric indices
        train_labels_idx = [label_map[label] for label in train_labels]
        val_labels_idx = [label_map[label] for label in val_labels]
        all_classes = list(label_map.values())  # Ensure all labels are considered in loss calculation

        # Convert labels to one-hot encoding for log loss computation
        lb = LabelBinarizer()
        lb.fit(all_classes)
        train_labels_one_hot = lb.transform(train_labels_idx)
        val_labels_one_hot = lb.transform(val_labels_idx)

        with TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "fasttext_train.txt")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                for text, label in zip(train_texts, train_labels):
                    f.write(f"{label} {text}\n")

            # Learning Rate Finder
            def find_best_lr():
                losses = []
                lrs = np.logspace(np.log10(lr_min), np.log10(lr_max), num=lr_finder_epochs)

                for lr in lrs:
                    model = fasttext.train_supervised(temp_file_path, epoch=1, lr=lr, verbose=0)
                    predictions = [label_map.get(model.predict(text)[0][0], -1) for text in val_texts]
                    if -1 in predictions:
                        logging.warning("Some predictions did not match expected labels. Adjusting...")
                        predictions = [p if p != -1 else np.random.choice(all_classes) for p in predictions]

                    predictions_one_hot = lb.transform(predictions)
                    loss = log_loss(val_labels_one_hot, predictions_one_hot) if len(set(val_labels_idx)) > 1 else 0.0
                    losses.append(loss)

                    if len(losses) > 1 and losses[-1] > losses[-2] * 1.5:
                        logging.warning("Loss is diverging! Stopping LR Finder.")
                        break

                best_lr = lrs[np.argmin(losses)]
                logging.info(f"Best LR Found: {best_lr:.6f}")
                return best_lr

            best_lr = find_best_lr()

            # Training with Early Stopping
            patience_counter = 0
            best_score = float("inf") if monitor_metric == "loss" else 0.0
            best_model = None

            for epoch in range(train_epochs):
                model = fasttext.train_supervised(temp_file_path, epoch=epoch + 1, lr=best_lr, verbose=0)
                train_predictions = [label_map.get(model.predict(text)[0][0], -1) for text in train_texts]
                val_predictions = [label_map.get(model.predict(text)[0][0], -1) for text in val_texts]

                if -1 in train_predictions or -1 in val_predictions:
                    logging.warning("Some predictions did not match expected labels. Adjusting...")
                    train_predictions = [p if p != -1 else np.random.choice(all_classes) for p in train_predictions]
                    val_predictions = [p if p != -1 else np.random.choice(all_classes) for p in val_predictions]

                train_accuracy = accuracy_score(train_labels_idx, train_predictions)
                val_accuracy = accuracy_score(val_labels_idx, val_predictions)

                train_predictions_one_hot = lb.transform(train_predictions)
                val_predictions_one_hot = lb.transform(val_predictions)

                train_loss = log_loss(train_labels_one_hot, train_predictions_one_hot) if len(
                    set(train_labels_idx)) > 1 else 0.0
                val_loss = log_loss(val_labels_one_hot, val_predictions_one_hot) if len(
                    set(val_labels_idx)) > 1 else 0.0

                current_score = val_loss if monitor_metric == "loss" else val_accuracy
                improvement = (best_score - current_score) if monitor_metric == "loss" else (current_score - best_score)

                logging.info(
                    f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2%} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%}")

                if improvement > min_delta:
                    best_score = current_score
                    best_model = model
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info("Early stopping triggered.")
                        break

            self.model = best_model

            # Prepare outputs
            y_true = val_labels_idx
            y_pred = [label_map.get(best_model.predict(text)[0][0], -1) for text in val_texts]
            y_pred_probabilities = [best_model.predict(text, k=len(unique_labels))[1] for text in val_texts]

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

    @time_operation
    def predict(self, test_inputs, test_labels=None, batch_size=16):
        """
        Predict labels for preprocessed test inputs and optionally compute accuracy and loss.

        Args:
            test_inputs: Input data (raw text for FastText, tokenized format for Transformers).
            test_labels (list, optional): Ground truth labels for evaluation.

        Returns:
            Numpy Array: Predicted labels.

        Raises:
            RuntimeError: If prediction fails.
        """
        try:
            if self.model_enum == LanguageModelsEnum.FASTTEXT:
                # Get predictions and probabilities
                predictions = [self.model.predict(text, k=1)[0][0] for text in test_inputs]
                probabilities = [list(self.model.predict(text, k=self.num_classes)[1]) for text in test_inputs]
            else:
                # Ensure the model is on the correct device
                self.model.to(self.device)
                # Convert inputs to tensors and move to device
                test_inputs = {key: torch.tensor(val).to(self.device) for key, val in test_inputs.items()}
                # Perform inference in batches
                predictions = []
                probabilities = []
                for i in range(0, len(test_inputs['input_ids']), batch_size):
                    batch_inputs = {key: val[i:i + batch_size] for key, val in test_inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**batch_inputs)
                    logits = outputs.logits
                    batch_probabilities = logits.softmax(dim=-1).tolist()
                    batch_predictions = logits.argmax(dim=-1).tolist()
                    predictions.extend(batch_predictions)
                    probabilities.extend(batch_probabilities)

            self.predicted_labels = predictions

            # If ground truth labels are provided, calculate accuracy and loss
            if test_labels is not None:
                if self.model_enum == LanguageModelsEnum.FASTTEXT:
                    label_map = {f"__label__{i}": i for i in range(self.num_classes)}
                    test_labels_idx = [label_map[label] for label in test_labels]
                    predictions_idx = [label_map[pred] for pred in predictions]
                else:
                    test_labels_idx = test_labels
                    predictions_idx = predictions

                # Compute accuracy
                accuracy = accuracy_score(test_labels_idx, predictions_idx)

                # Compute log loss (only if more than one class exists)
                loss = log_loss(test_labels_idx, probabilities, labels=list(range(self.num_classes))) if len(
                    set(test_labels_idx)) > 1 else 0.0

                self.testing_metrics = {"loss": loss, "accuracy": accuracy}
                self.ground_truth = test_labels

                if self.model_enum == LanguageModelsEnum.FASTTEXT:
                    self.ground_truth = [int(re.search(r'\d+', label).group()) for label in self.ground_truth]
                    predictions = [int(re.search(r'\d+', label).group()) for label in predictions]

                self.predicted_labels = predictions

                return np.array(predictions)

            if self.model_enum == LanguageModelsEnum.FASTTEXT:
                self.ground_truth = [int(re.search(r'\d+', label).group()) for label in self.ground_truth]
                predictions = [int(re.search(r'\d+', label).group()) for label in predictions]

            self.predicted_labels = predictions

            return np.array(predictions)

        except Exception as e:
            raise RuntimeError(f"Error during prediction: {e}")

    @time_operation
    def predict_prob(self, test_inputs, batch_size=16):
        """
        Predict probabilities for the given inputs in batches.

        Args:
            test_inputs: Input data (text for FastText, tokenized format for Transformers).
            batch_size (int): Batch size for prediction.

        Returns:
            Numpy Array: Predicted probabilities.

        Raises:
            RuntimeError: If probability prediction fails.
        """
        try:
            if self.model_enum == LanguageModelsEnum.FASTTEXT:
                # FastText handles probabilities directly
                self.prediction_probabilities = [list(self.model.predict(text, k=self.num_classes)[1]) for text in test_inputs]
            else:
                # Ensure the model is on the correct device
                self.model.to(self.device)

                # Convert inputs to tensors and move to device
                test_inputs = {key: torch.tensor(val, dtype=torch.long).to(self.device) for key, val in test_inputs.items()}

                # Perform inference in batches
                probabilities = []
                for i in range(0, len(test_inputs['input_ids']), batch_size):
                    batch_inputs = {key: val[i:i + batch_size] for key, val in test_inputs.items()}
                    with torch.no_grad():
                        outputs = self.model(**batch_inputs)
                    batch_probabilities = outputs.logits.softmax(dim=-1).tolist()
                    probabilities.extend(batch_probabilities)

                self.prediction_probabilities = probabilities

            return np.array(self.prediction_probabilities)

        except Exception as e:
            raise RuntimeError(f"Error during probability prediction: {e}")

    def save_model(self, filepath):
        """
        Save the model along with metadata.

        Args:
            filepath (str): Directory where the model and metadata will be saved.

        Raises:
            RuntimeError: If there is an error while saving the model.
        """
        try:
            os.makedirs(filepath, exist_ok=True)
            metadata = {
                "model_type": self.model_enum.name,
                "num_classes": self.num_classes
            }

            if self.model_enum == LanguageModelsEnum.FASTTEXT:
                model_path = os.path.join(filepath, "fasttext_model.bin")
                self.model.save_model(model_path)
                metadata["fasttext_model_path"] = model_path  # Store path in metadata

            else:
                self.model.save_pretrained(filepath)

            metadata_path = os.path.join(filepath, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as meta_file:
                json.dump(metadata, meta_file, indent=4)

        except Exception as e:
            raise RuntimeError(f"Error saving the model: {e}")

    def load_model(self, filepath):
        """
        Load the model along with metadata.

        Args:
            filepath (str): Directory where the model and metadata are stored.

        Raises:
            RuntimeError: If the model file is missing or an error occurs while loading.
        """
        try:
            metadata_path = os.path.join(filepath, "metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError("Metadata file not found in the specified path.")

            with open(metadata_path, "r", encoding="utf-8") as meta_file:
                metadata = json.load(meta_file)

            self.model_enum = LanguageModelsEnum[metadata["model_type"]]
            self.num_classes = metadata["num_classes"]

            if self.model_enum == LanguageModelsEnum.FASTTEXT:
                fasttext_path = metadata.get("fasttext_model_path", os.path.join(filepath, "fasttext_model.bin"))
                if not os.path.exists(fasttext_path):
                    raise FileNotFoundError(f"FastText model not found at {fasttext_path}.")
                self.model = fasttext.load_model(fasttext_path)

            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(filepath)
                self.model.to(self.device)

        except FileNotFoundError as e:
            raise RuntimeError(f"File error while loading model: {e}")

        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error reading metadata JSON: {e}")

        except Exception as e:
            raise RuntimeError(f"Error loading the model: {e}")
