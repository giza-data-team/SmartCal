Meta-Model Training
## Overview

This directory contains the code for training multi-class classification models to predict the best calibration method for different datasets and metrics. The system processes meta-data and trains AdaBoost classifiers for each calibration metric separately.

## About the Approach

The training script implements a metric-specific multi-class classification strategy where:
- For each unique calibration metric in the dataset, a separate multi-class classifier is trained
- Each classifier learns to predict the best calibration method among all available options
- The system uses AdaBoost with Decision Tree base estimators for robust performance
- Models are evaluated using both standard metrics and ranking-based metrics (Recall@N and MRR)

This approach allows for metric-specific calibration recommendations, accounting for the fact that different metrics may favor different calibration methods.

## Training Script

The `train_and_process_meta_models.py` script:
1. Loads meta-data from the configured path
2. Preprocesses features using one-hot encoding for categorical variables
3. For each calibration metric:
   - Creates a metric-specific dataset
   - Splits data into train/test sets (90%/10%)
   - Trains an AdaBoost classifier with optimized hyperparameters
   - Evaluates using multiple metrics including Recall@N and MRR
   - Saves the trained model and label encoder
   - Generates comprehensive insights and visualizations

## Model Architecture

**AdaBoost Classifier Configuration:**
- Base estimator: Decision Tree with max_depth=3, min_samples_split=3, min_samples_leaf=1
- Number of estimators: 250
- Learning rate: 0.08
- Criterion: Gini impurity

## Evaluation Metrics

The system evaluates models using:
- **Ranking Metrics:** Recall@5, Mean Reciprocal Rank (MRR)
- **Standard Metrics:** Accuracy, F1-score (micro, macro, weighted), Precision, Recall

## Model Storage

All models and artifacts are saved in metric-specific subdirectories under `package/src/smartcal/config/resources/models/`:
- `{metric_name}/AdaBoost.joblib`: Trained classifier for each metric
- `{metric_name}/label_encoder.joblib`: Label encoder for converting between numeric and original class labels

The models are saved to the package resources directory to be accessible by the MetaModel class for inference.

## Insights Generation

The system automatically generates comprehensive visualizations:
- **Class Distribution Plots:** Bar charts showing calibration method frequency per metric
- **Feature Correlation Heatmaps:** Correlation matrices for numerical features
- **Pairwise Ratio Heatmaps:** Outperformance ratios between calibration methods
- **Performance Analysis:** CSV files with detailed evaluation metrics

All insights are saved in the `meta_models_training/insights/` directory.

## Configuration

The training uses the following configuration:
- Test split: 10% of data
- Random state: 42
- Target column: 'Best_Cal'
- N value for Recall@N: 5
- Model storage path: `package/src/smartcal/config/resources/models/`
- Insights path: `meta_models_training/insights/`

## Usage

To train the models and generate insights, run from root directory:
```bash
python -m meta_models_training.train_and_process_meta_models
```

## Output Structure

```
package/src/smartcal/config/resources/models/
├── ECE/
│   ├── AdaBoost.joblib
│   └── label_encoder.joblib
├── MCE/
│   ├── AdaBoost.joblib
│   └── label_encoder.joblib
├── ConfECE/
│   ├── AdaBoost.joblib
│   └── label_encoder.joblib
├── brier_score/
│   ├── AdaBoost.joblib
│   └── label_encoder.joblib
└── log_loss/
    ├── AdaBoost.joblib
    └── label_encoder.joblib

meta_models_training/insights/
├── class_distributions/
├── feature_correlation_heatmaps/
├── pairwise_ratio_heatmaps/
└── model_performance_analysis/
```