# End-to-End Evaluation (`/experiments/end_to_end_eval`)

This directory contains scripts and documentation for benchmarking the performance of different calibration approaches.

## Overview

This evaluation suite compares three different approaches to model calibration:

1. **SmartCal End-to-End**: Meta-model assisted calibration with Bayesian Optimization
2. **Random Search**: Random search over calibration algorithms and hyperparameters
3. **Fixed Calibrators**: Evaluation of fixed calibration algorithms with default hyperparameters
## Approaches

### 1. AutoCal End-to-End (Meta-Model + Bayesian Optimization)

**Objective**: Optimize calibration using a meta-model to recommend calibration methods, followed by iteration-constrained Bayesian Optimization (BO).

**Steps**:
1. **Load Data**: Split data into train (train classifiers), cal (calibration tuning), and test (final evaluation).
2. **Train Classification Models**: Train base classifiers on the train set.
3. **Meta-Model Recommendations**:
   - Use a pre-trained meta-model to recommend K = 5 calibration algorithms (e.g., Beta Calibration, Temperature Scaling).
   - Normalize the meta-model's confidence scores to allocate iterations proportionally for each recommendation.
4. **Bayesian Optimization**:
   - Run BO on the cal set for 10/30/50 total iterations (allocated based on confidence scores).
   - Optimize hyperparameters to minimize ECE using K-Fold Cross-Validation on the calibration set.
5. **Select Best Calibrator**: Choose the method with the lowest ECE on the cal set.
6. **Evaluate**: Apply the best calibrator to the test set.
7. **Save Results**:
   - Store LogLoss, F1, ECE, ECE conf, MCE, Reliability Diagram, Brier score, Time for training, calibration, and test sets.
   - Save hyperparameters, iteration allocation details, and optimization history.

**Usage**:

To run the script from root directory:
```bash
python -m experiments.end_to_end_eval.smartcal_end_to_end_script
```

**Output**:
- Results are saved to `experiments/end_to_end_eval/Results/smartcal_results.csv`
- The CSV contains detailed metrics for each dataset/model combination across different iteration counts


### 2. Random Search Over Search Space

**Objective**: Compare against a brute-force random search over all calibration algorithms/hyperparameters.

**Steps**:
1. **Load Data**: Split data into train, cal, and test sets.
2. **Train Classifier**: Train the same base classifiers on the train set.
3. **Define Search Space**: Include all calibration algorithms and their discrete hyperparameters.
4. **Random Search Execution**:
   - Randomly sample 10/30/50 configurations (algorithm + hyperparameters).
   - Evaluate each on the cal set using K-Fold Cross-Validation and ECE.
5. **Select Best Configuration**: Choose the configuration with the lowest ECE on cal.
6. **Evaluate**: Test the best configuration on the test set.
7. **Save Results**:
   - Record LogLoss, F1, ECE, ECE conf, MCE, Reliability Diagram, Brier score, Time for training, calibration, and test sets.
   - Save sampled configurations and optimization history.

**Usage**:

To run the script from root directory:

```bash
python -m experiments.end_to_end_eval.baseline_random_search_script
```

**Output**:
- Results are saved to `experiments/end_to_end_eval/Results/baseline_random_search_results.csv`
- The CSV contains metrics for randomly sampled calibration algorithm configurations

### 3. Fixed Calibrators with Default Hyperparameters

**Objective**: Evaluate fixed calibrators (Beta Calibration, Temperature Scaling) using default hyperparameters.

**Steps**:
1. **Load Data**: Split data into train, cal, and test sets.
2. **Train Classifier**: Train base models on the train set.
3. **Fixed Calibration Algorithms**:
   - Beta Calibration (default hyperparameters).
   - Temperature Scaling (default hyperparameters).
4. **Calibration Tuning**:
   - Use K-Fold Cross-Validation on the calibration set to evaluate ECE for each calibrator.
   - No hyperparameter tuning; use default settings.
5. **Fit Calibrators**:
   - Fit each calibrator on the full cal set using default hyperparameters.
6. **Evaluate**: Test both calibrators on the test set.
7. **Repeat and Save Results**:
   - Repeat the process 5 times with randomized data splits by using different seeds (26, 30, 42, 78, 101).
   - Report mean and standard deviation of LogLoss, F1, ECE, ECE conf, MCE, Reliability Diagram, Brier score, Time across trials for each calibrator.

**Usage**:


First, generate results by running the following script from root directory
```bash
python -m experiments.end_to_end_eval.csv_experiments_script
```

Then, run the aggregation script:
```bash
python -m experiments.end_to_end_eval.dump_calibrator_script
```
**Output**:
- Aggregated Results are saved to `experiments/end_to_end_eval/Results/dump_calibrator_results_summary.csv`
- The summary file includes means and standard deviations for the metrics grouped by dataset, classifier, and calibrator