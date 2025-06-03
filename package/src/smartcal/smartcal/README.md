# SmartCal Class

The `SmartCal` class provides an automated, meta-learning-based framework for **probabilistic model calibration**. It intelligently selects and tunes the best calibration method based on extracted meta-features and predicted probabilities.

---

## Class Overview

**File**: `smartcal.py`  
**Main Class**: `SmartCal`

This class performs:

- Meta-feature extraction from predictions and labels
- Recommending calibration algorithms using a meta-model
- Hyperparameter tuning using Bayesian Optimization
- Selecting the best calibration method based on calibration metrics (supports **ECE**, **MCE**, **ConfECE**, **brier_score**, and **log_loss**)

---

## Main Methods

### `recommend_calibrators(y_true, predictions_prob, n=5)`

Recommends the top-N calibration methods based on input labels and prediction probabilities.

**Parameters:**
- `y_true`: Ground truth labels (int or one-hot).
- `predictions_prob`: Model output probabilities (NumPy array).
- `n`: Number of calibrators to recommend (1 to 12).

**Returns:**  
`List[Tuple[str, float]]` — List of recommended calibrator names and normalized confidence scores.

---

### `best_fitted_calibrator(y_true, predictions_prob, n_iter=10)`

Fits the best calibration model by tuning each recommended calibrator with Bayesian optimization.

**Parameters:**
- `y_true`: Ground truth labels.
- `predictions_prob`: Predicted probabilities to be calibrated.
- `n_iter`: Total number of optimization iterations to allocate.

**Returns:**  
A fitted calibrator object (must have `.predict()` and `.fit()` methods).

---

## Example Workflow

```python
from smartcal import SmartCal

# Initialize SmartCal with desired metric
smartcal = SmartCal(metric='ECE')  # Can use 'MCE', 'ConfECE', 'brier_score', or 'log_loss'

# Step 1: Get top 3 recommended calibration methods
recommended = smartcal.recommend_calibrators(y_true, predictions_prob, n=3)

# Step 2: Fit and retrieve the best calibrator
best_calibrator = smartcal.best_fitted_calibrator(y_true, predictions_prob, n_iter=20)

# Step 3: Use the calibrator
calibrated_probs = best_calibrator.predict(predictions_prob)
```

---

## ⚠️ Errors and Validations

- Checks for valid range of `n` (1–12).
- Validates array shapes and presence of NaN/Inf.
- Validates that the specified metric is supported.
- Logs failures during optimization/fitting using Python `logging`.

---

## Dependencies

Make sure the following components exist and are functional:

- `MetaFeaturesExtractor` (`meta_features_extraction/`)
- `MetaModel` (`meta_model/`)
- `CalibrationOptimizer` (`bayesian_optimization/`)
- `CalibrationAlgorithmTypesEnum` (`config/enums/`)
- `compute_calibration_metrics`, `convert_one_hot_to_labels` (`utils/`)
- `ConfigurationManager` (`config/configuration_manager/`)

---

## Testing

You should test `SmartCal` using datasets that:
- Include class probabilities (`predictions_prob`)
- Use either one-hot or integer-encoded ground truth labels (`y_true`)

---