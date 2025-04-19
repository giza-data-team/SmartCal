Meta-Model (`/meta_model_training`)
## Overview

This directory contains the code for training binary classification models using a one-vs-all approach. Each model is trained to identify a specific class against all others.

## About the One-vs-All Approach

The training script implements a one-vs-all (also known as one-vs-rest) classification strategy where:
- For each unique class in the target variable, a separate binary classifier is trained
- Each classifier learns to distinguish one class (positive) from all other classes (negative)
- During prediction, all classifiers evaluate a new instance, and the class with the highest confidence score is selected

This approach is particularly useful for multi-class problems where class imbalance exists.

## Training Script

The `train_meta_model.py` script:
1. Loads meta-data
2. Preprocesses features (one-hot encoding) and target variables
3. For each class:
   - Converts the problem into a binary classification task
   - Applies random undersampling to balance the classes
   - Trains a logistic regression model
   - Evaluates using F1-score
   - Saves the trained model to disk

## Model Storage

All models and necessary artifacts are saved in the `meta_model_training/Results/` directory:
- `lr_ova_model.joblib`: One vs All model
- `label_encoder.joblib`: Encoder to convert between numeric and original class labels

## Configuration

The training uses the following configuration:
- Test split: 10% of data
- Target column: 'Best_Cal'
- Logistic Regression parameters:
  - max_iter: 5000
  - C: 0.1
  - penalty: l2
  - solver: newton-cg

## Usage

To train the models, run from root dir:
```bash
python -m meta_model_training.train_meta_model
```