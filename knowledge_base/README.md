# Knowledge Base (`/knowledge_base`)

## Overview
The knowledge base is structured to store metadata for datasets used in binary and multiclass classification tasks.
It contains information regarding dataset characteristics, classification models, calibration algorithms, and performance metrics
## Schema

Each entry in the knowledge base includes the following information:

### Basic Information

- **id**: Unique identifier for the experiment
- **dataset_name**: Name of the dataset used
- **no_classes**: Number of classes in the classification problem
- **no_instances**: Total number of instances in the dataset
- **problem_type**: Type of problem (e.g., classification)
- **classification_type**: Binary or multiclass classification
- **classification_model**: Model used for classification
- **calibration_algorithm**: Algorithm used for calibration
- **cal_hyperparameters**: Hyperparameters used for calibration
- **status**: Current status of the experiment
- **created_at**: Timestamp when the experiment was created
- **updated_at**: Timestamp when the experiment was last updated

### Dataset Information

- **n_instances_cal_set**: Number of instances in the calibration set
- **split_ratios_train**: Ratio of data used for training
- **split_ratios_cal**: Ratio of data used for calibration
- **split_ratios_test**: Ratio of data used for testing

### Timing Information

- **preprocessing_fit_time**: Time taken to fit preprocessing
- **preprocessing_transform_time**: Time taken to transform data
- **train_time**: Time taken to train the model
- **test_time**: Time taken to test the model
- **calibration_fit_time**: Time taken to fit calibration
- **calibration_predict_time**: Time taken for calibration prediction

### Uncalibrated Model Performance

Metrics for train, calibration, and test sets including:

- Loss, accuracy, ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error), conf_ece (Confidence ECE)
- F1 scores (macro, micro, weighted)
- Recall and precision metrics
- Brier score
- Calibration curves

### Calibrated Model Performance

Metrics for calibration and test sets after applying calibration, including:

- Loss, accuracy, ECE
- MCE, conf_ece
- Brier score
- Calibration curves

### Additional Information

- **ground_truth_test_set**: True labels for the test set
- **ground_truth_cal_set**: True labels for the calibration set
- **error_message**: Any error messages encountered during processing
- Calibration bin information for different dataset splits

## Dataset Types

The knowledge base includes experiments on three types of datasets:

1. **Tabular Datasets**: 160 datasets representing structured data
2. **Language Datasets**: 2 datasets for text classification
3. **Image Datasets**: 3 datasets for image classification

Each dataset type has unique characteristics and challenges, allowing for comprehensive analysis of calibration techniques across different data modalities.

## Usage

### Running Experiments

The knowledge base is constructed by running experiments through the `run_knowledge_base_experiments_script.py`.

To run from root directory:

```bash
python knowledge_base/run_knowledge_base_experiments_script.py
```

### Dataset Type Selection

To run experiments for different dataset types, modify the `DatasetTypesEnum` parameter:

1. For tabular datasets:
```python
exp_manager = ExperimentManager(DatasetTypesEnum.TABULAR, ExperimentType.KNOWLEDGE_BASE_V2)
```

2. For language datasets:
```python
exp_manager = ExperimentManager(DatasetTypesEnum.LANGUAGE, ExperimentType.KNOWLEDGE_BASE_V2)
```

3. For image datasets:
```python
exp_manager = ExperimentManager(DatasetTypesEnum.IMAGE, ExperimentType.KNOWLEDGE_BASE_V2)