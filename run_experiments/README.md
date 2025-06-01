# Experiments

## Overview

The experiments system is structured to store metadata for datasets used in binary and multiclass classification tasks.
It contains information regarding dataset characteristics, classification models, calibration algorithms, and performance metrics.
The experiments system was constructed through a systematic offline process to evaluate calibration algorithms across diverse datasets.

## Experiment Construction Approach

### Data Collection & Preparation
- 195 datasets from Kaggle, UCI, OpenML, and torchvision across tabular, language, and image domains
- **Knowledge Base Experiments**: 165 datasets (160 tabular + 2 language + 3 image)
- **Benchmarking Experiments**: 30 datasets (26 tabular + 2 language + 2 image)
- Split ratio: 60% training, 20% calibration, 20% test

### Model Training & Calibration
- Applied domain-specific classification algorithms to each dataset
- Implemented 12 post-hoc calibration algorithms with hyperparameter grid search
- Evaluated using 5 metrics: MCE, ECE, Confidence ECE, Brier Score, and Log Loss

### Experiment Database Population
- For each dataset-classifier pair, recorded the best calibration algorithm per metric
- Stored comprehensive metadata including dataset characteristics, model information, hyperparameters, and performance metrics
- Tracked timing information for all processing steps

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

The experiments include three types of datasets with the following distribution:

1. **Tabular Datasets**: 186 total datasets
   - Knowledge Base: 160 datasets
   - Benchmarking: 26 datasets

2. **Language Datasets**: 4 total datasets  
   - Knowledge Base: 2 datasets
   - Benchmarking: 2 datasets

3. **Image Datasets**: 5 total datasets (from torchvision)
   - Knowledge Base: 3 datasets
   - Benchmarking: 2 datasets

Each dataset type has unique characteristics and challenges, allowing for comprehensive analysis of calibration techniques across different data modalities.

## Calibration Algorithms

The system supports 12 calibration algorithms:

1. **Empirical Binning** - Histogram-based binning calibration
2. **Isotonic Regression** - Non-parametric calibration method
3. **Beta Calibration** - Parametric method using Beta distribution
4. **Temperature Scaling** - Simple parametric scaling method
5. **Dirichlet Calibration** - Multi-class extension of Platt scaling
6. **Meta Calibration** - Ensemble method combining multiple calibrators
7. **Matrix Scaling** - Affine transformation for multi-class calibration
8. **Vector Scaling** - Extension of temperature scaling
9. **Platt Scaling** - Sigmoid-based calibration method
10. **Histogram Calibration** - Advanced histogram-based method
11. **Mix and Match** - Adaptive combination of parametric and non-parametric methods
12. **Adaptive Temperature Scaling** - Dynamic temperature adjustment

## Usage

### Running Experiments

The experiments are executed by running the experiment script through the `run_experiments_script.py`.

To run from root directory:

```bash
python -m run_experiments.run_experiments_script
```

### Dataset Type Selection

To run experiments for different dataset types, use the following line of code in `run_experiments_script.py` and change the enum options:

```python
exp_manager = ExperimentManager(DatasetTypesEnum.OPTION, ExperimentType.OPTION)
```

Replace the first `OPTION` with one of:
- `TABULAR` - for tabular datasets
- `LANGUAGE` - for language datasets  
- `IMAGE` - for image datasets

Replace the second `OPTION` with one of:
- `KNOWLEDGE_BASE_V2` - for building the knowledge base (165 datasets)
- `BENCHMARKING_V2` - for benchmarking approaches (30 datasets)

Example:
```python
exp_manager = ExperimentManager(DatasetTypesEnum.TABULAR, ExperimentType.KNOWLEDGE_BASE_V2)
```

### Experiment Types

The system supports different experiment types through the `ExperimentType` enum:

- `KNOWLEDGE_BASE_V2`: For building the knowledge base with comprehensive calibration algorithm evaluation (165 datasets)
- `BENCHMARKING_V2`: For benchmarking specific calibration approaches (30 datasets)

### Experiment Management

The experiments are managed through the `ExperimentManager` class located in `experiment_manager/experiment_manager.py`. This class handles:

- Experiment configuration generation
- Database operations
- Experiment execution coordination
- Status tracking and error handling