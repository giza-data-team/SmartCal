# End-to-End Evaluation

This directory contains scripts and documentation for benchmarking the performance of different calibration approaches in the AutoML Calibrator framework.

## Overview

This evaluation suite provides a comprehensive comparison of three distinct approaches to model calibration:

1. **SmartCal End-to-End**: Meta-model assisted calibration with Bayesian Optimization
2. **Random Search Baseline**: Random search over calibration algorithms and hyperparameters  
3. **Fixed Calibrators Baseline**: Evaluation of fixed calibration algorithms with default hyperparameters

## Directory Structure

```
experiments/end_to_end_eval/
├── README.md                           # This documentation
├── smartcal_end_to_end_script.py      # SmartCal approach implementation
├── baseline_random_search_script.py   # Random search baseline
├── csv_experiments_script.py          # Fixed calibrators evaluation
├── dump_calibrator_script.py          # Results aggregation
└── Results/                           # Output directory for results
    ├── smartcal_results.csv
    ├── baseline_random_search_results.csv
    └── dump_calibrator_results_summary.csv
```

## Evaluation Approaches

### 1. SmartCal End-to-End (Meta-Model + Bayesian Optimization)

**Objective**: Optimize calibration using a meta-model to recommend calibration methods, followed by iteration-constrained Bayesian Optimization.

**Methodology**:
1. **Data Preparation**: Split datasets into train (classifier training), cal (calibration tuning), and test (final evaluation) sets
2. **Base Model Training**: Train classification models on the training set
3. **Meta-Model Recommendations**: 
   - Leverage pre-trained meta-model to recommend top K=5 calibration algorithms
   - Examples: Beta Calibration, Temperature Scaling, Platt Scaling
   - Normalize confidence scores to proportionally allocate optimization iterations
4. **Bayesian Optimization**:
   - Execute BO on calibration set with 10/30/50 total iterations
   - Distribute iterations based on meta-model confidence scores
   - Optimize hyperparameters to minimize calibration metrics (ECE, MCE, ConfECE, log_loss, Brier score)
   - Use K-Fold Cross-Validation for robust evaluation
5. **Model Selection**: Select calibrator with lowest validation metrics on calibration set
6. **Final Evaluation**: Apply best calibrator to test set
7. **Results Storage**: Save comprehensive metrics, hyperparameters, and optimization history

**Usage**:
```bash
python -m experiments.end_to_end_eval.smartcal_end_to_end_script
```

**Output**:
- Results saved to: `experiments/end_to_end_eval/Results/smartcal_results.csv`
- Contains detailed metrics for each dataset/model combination across different iteration budgets

### 2. Random Search Baseline

**Objective**: Establish baseline performance through brute-force random search over the complete calibration algorithm and hyperparameter space.

**Methodology**:
1. **Data Preparation**: Identical train/cal/test split as SmartCal approach
2. **Base Model Training**: Train same classification models on training set
3. **Search Space Definition**: Include all available calibration algorithms with their hyperparameter ranges
4. **Random Sampling**:
   - Randomly sample 10/30/50 algorithm-hyperparameter configurations
   - Evaluate each configuration on calibration set using K-Fold Cross-Validation
   - Compute calibration metrics (ECE, MCE, ConfECE, log_loss, Brier score)
5. **Best Configuration Selection**: Choose configuration with optimal validation metrics
6. **Final Evaluation**: Test selected configuration on test set
7. **Results Storage**: Record all metrics, sampled configurations, and search history

**Usage**:
```bash
python -m experiments.end_to_end_eval.baseline_random_search_script
```

**Output**:
- Results saved to: `experiments/end_to_end_eval/Results/baseline_random_search_results.csv`
- Contains performance metrics for randomly sampled calibration configurations

### 3. Fixed Calibrators Baseline

**Objective**: Evaluate standard calibration methods using default hyperparameters to establish fundamental performance baselines.

**Methodology**:
1. **Data Preparation**: Consistent train/cal/test split methodology
2. **Base Model Training**: Train classification models on training set
3. **Fixed Algorithm Evaluation**:
   - Beta Calibration with default hyperparameters
   - Temperature Scaling with default hyperparameters
4. **Validation Process**:
   - Apply K-Fold Cross-Validation on calibration set
   - No hyperparameter optimization - use default settings only
5. **Calibrator Training**: Fit each calibrator on full calibration set
6. **Test Evaluation**: Apply calibrators to test set
7. **Statistical Robustness**:
   - Repeat entire process 5 times with different random seeds (26, 30, 42, 78, 101)
   - Compute mean and standard deviation across trials
   - Report statistical significance of results

**Usage**:

Step 1 - Generate individual results:
```bash
python -m experiments.end_to_end_eval.csv_experiments_script
```

Step 2 - Aggregate and summarize results:
```bash
python -m experiments.end_to_end_eval.dump_calibrator_script
```

**Output**:
- Aggregated results saved to: `experiments/end_to_end_eval/Results/dump_calibrator_results_summary.csv`
- Summary includes means and standard deviations for all metrics, grouped by dataset, classifier, and calibrator

## Evaluation Metrics

All approaches evaluate the following calibration and classification metrics:

- **Calibration Metrics**:
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE) 
  - Confidence-based ECE (ConfECE)
  - Brier Score
- **Classification Metrics**:
  - Log Loss
  - F1 Score
- **Additional Outputs**:
  - Reliability Diagrams
  - Training/Calibration/Test execution times