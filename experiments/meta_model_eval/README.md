# Meta-Model Evaluation

This directory contains scripts for evaluating the performance of meta-models in selecting the best calibration algorithm for classification tasks. It compares the performance of the meta-model against a random selection baseline to assess the quality of the predictions.

## Scripts

### 1. Extract Meta-Model Results (`meta_model_results_extraction_script.py`)

#### Purpose
Generates and evaluates meta-model predictions for selecting the best calibration algorithms across different datasets and classification models. The predictions are compared against actual performance benchmarks.

#### Key Features
- **Extracts Meta-Features**: Computes meta-features for each dataset-model pair using true and predicted labels from calibration sets.
- **Generates Top-N Calibrator Predictions**: The meta-model predicts the top `N` calibration algorithms based on the meta-features.
- **Evaluates Predictions**: Compares predicted calibrators against actual performance metrics like ECE, Brier score, and loss.
- **Multiple Evaluation Metrics Supported**: Supports various evaluation metrics to measure calibrator performance.

#### Usage
To run the meta-model evaluation script and generate results from the root directory:

```bash
python -m experiments.meta_model_eval.meta_model_results_extraction_script
```

#### Output Path
Results are saved to: `experiments/meta_model_eval/Results/meta_model_results.csv`

#### Output Format
The script generates a CSV file with the following columns:

| **Column Name**            | **Description**                                                   |
|----------------------------|-------------------------------------------------------------------|
| `Dataset_Name`             | Unique identifier for the dataset.                               |
| `Dataset_Type`             | Type of dataset (e.g., binary or multi-class).                   |
| `Problem_Type`             | Problem type (e.g., tabular, image, or language).                |
| `Model_Type`               | Classification algorithm used for prediction.                     |
| `N`                        | The number of top predictions (calibrators) considered.           |
| `Predicted_Calibrators`    | List of predicted calibrators (comma-separated).                 |
| `Evaluation_Metric`        | Type of evaluation metric used (e.g., ECE, Brier score, Loss).   |
| `Best_Calibrator`          | The calibrator with the best performance based on the evaluation metric. |
| `Best_Performance_Value`   | Performance score of the best calibrator (e.g., lower ECE, Brier score, or loss). |
| `Meta_Model_Confidence`    | Confidence scores of the meta-model's top recommendations.       |
| `Meta_Model_Version`       | Version of the meta-model used.                                  |
| `Is_Best_Calibrator`       | Boolean indicating if the selected calibrator is the best based on the database for the given metric. |

### 2. Extract Random Selection Results (`baseline_results_extraction_script.py`)

#### Purpose
Implements a random selection baseline for comparing the performance of the meta-model against a purely random strategy with reproducible results.

#### Key Features
- **Random Selection of Calibrators**: Selects `N` random calibration algorithms from the benchmarking table for each dataset-model pair.
- **Consistent Seed Generation**: Ensures reproducibility by generating a consistent random seed based on the dataset and model name.
- **Multiple Metric Evaluation**: Evaluates the randomly selected calibrators using metrics like ECE, Brier score, and loss.
- **Performance Comparison**: Allows for a direct comparison between the random selection strategy and meta-model predictions.

#### Usage
To run the random selection baseline and generate results from the root directory:

```bash
python -m experiments.meta_model_eval.baseline_results_extraction_script
```

#### Output Path
Results are saved to: `experiments/meta_model_eval/Results/baseline_results.csv`

#### Output Format
The output CSV file contains the following columns:

| **Column Name**           | **Description**                                                   |
|---------------------------|-------------------------------------------------------------------|
| `Dataset_Name`            | Unique identifier for the dataset.                               |
| `Dataset_Type`            | Type of dataset (e.g., binary or multi-class).                   |
| `Problem_Type`            | Problem type (e.g., tabular, image, or language).               |
| `Model_Type`              | Classification algorithm used for prediction.                     |
| `N`                        | The number of random calibrators selected.                       |
| `Selected_Calibrators`    | List of randomly selected calibrators (comma-separated).         |
| `Evaluation_Metric`       | The evaluation metric used (e.g., ECE, Brier score, Loss).       |
| `Best_Calibrator`         | The calibrator with the best performance based on the evaluation metric. |
| `Best_Performance_Value`  | Performance score of the best calibrator.                        |

---

## Common Components

### Evaluation Metrics
The following metrics are used for evaluating calibrators:
- **Expected Calibration Error (ECE)**: A measure of calibration quality.
- **Brier Score**: Measures the mean squared difference between predicted probabilities and actual outcomes.
- **Loss Function**: A log loss measure used for evaluating calibration quality.
