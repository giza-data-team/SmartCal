# Calibration Algorithms

The following calibration methods are available, each with their configurable parameters:

1. **Empirical Binning Calibration**
2. **Temperature Scaling Calibration**
3. **Isotonic Calibration**
4. **Beta Calibration**
5. **Dirichlet Calibration**
6. **Meta Calibration**
7. **Matrix Scaling Calibration**
8. **Vector Scaling Calibration**
9. **Platt Scaling Calibration**
10. **Histogram Calibration**
11. **Mix-and-Match Calibration**
12. **Adaptive Temperature Scaling Calibration**
13. **Probability Tree Calibration**

## Input/Output Specifications

All calibrators expect:

**Input (fit):**
- `predictions`: `(n_samples, n_classes)` probabilities (must be in range (0,1))
- `ground_truth`: `(n_samples,)` integer class labels (0 to n_classes-1)

**Input (predict):**
- Same format as fitting input for probabilities:
  `predictions`: `(n_samples, n_classes)` probabilities (must be in range (0,1))

**Output (predict):**
- Returns `(n_samples, n_classes)` calibrated probabilities

**Note:** The `ProbabilityTreeCalibrator` has a different interface:
- **Input (fit):** `X` (features), `logits` (predictions), `y` (ground truth)
- **Input (predict):** `X` (features), `logits` (predictions)

## Hyperparameter Descriptions and Default Values

1. **EmpiricalBinningCalibrator**
   - **`n_bins`**: Number of bins used for binning.
     - Default: `10`

2. **IsotonicCalibrator**
   - No hyperparameters available.
     - Default: `None`

3. **BetaCalibrator**
   - **`model_type`**: Type of Beta model from betacal: `"abm"`, `"am"`, or `"ab"`.
     - Default: `'abm'`

4. **TemperatureScalingCalibrator**
   - **`initial_T`**: Initial temperature for scaling.
     - Default: `1.0`
   - **`lr_tempscaling`**: Learning rate for optimization.
     - Default: `0.01`
   - **`max_iter_tempscaling`**: Maximum number of iterations for optimization.
     - Default: `100`

5. **VectorScalingCalibrator**
   - **`lr`**: Learning rate for optimization.
     - Default: `0.01`
   - **`max_iter`**: Maximum number of iterations for optimization.
     - Default: `100`

6. **MatrixScalingCalibrator**
   - **`lr`**: Learning rate for optimization.
     - Default: `0.01`
   - **`max_iter`**: Maximum number of iterations for optimization.
     - Default: `100`

7. **DirichletCalibrator**
   - **`lr`**: Learning rate for optimization.
     - Default: `0.01`
   - **`max_iter`**: Maximum number of iterations for optimization.
     - Default: `100`

8. **MetaCalibrator**
   - **`alpha`**: Regularization parameter.
     - Default: `0.1`
   - **`acc`**: Accuracy threshold for calibration.
     - Default: `0.85`
   - **`calibrator_type`**: Type of constraint applied: `'ALPHA'`, or `'ACC'`.
     - Default: `'ALPHA'`

9. **PlattCalibrator**
   - **`calibrator_type`**: Type of calibration used: `'PLATT'`, `'PLATTBINNER'`, `'PLATTBINNERMARGINAL'`.
     - Default: `'PLATT'`
   - **`n_bins`**: Number of bins used for binning.
     - Default: `10`
  

10. **HistogramCalibrator**
    - **`calibrator_type`**: Type of histogram calibration used: `'HISTOGRAM'`, `'HISTOGRAMMARGINAL'`.
      - Default: `'HISTOGRAM'`
    - **`n_bins`**: Number of bins used for binning.
      - Default: `10`


11. **AdaptiveTemperatureScalingCalibrator**
    - **`lr_tempscaling`**: Learning rate for optimization.
      - Default: `0.01`
    - **`max_iter_tempscaling`**: Maximum number of iterations for optimization.
      - Default: `100`
    - **`confidence_bins`**: Number of bins for confidence-based calibration.
      - Default: `10`
    - **`entropy_bins`**: Number of bins for entropy-based calibration.
      - Default: `10`
    - **`initial_T`**: Initial temperature for scaling.
      - Default: `1.0`
    - **`mode`**: Mode of temperature scaling: `'linear'`, `'entropy'`, `'hybrid'`.
      - Default: `'hybrid'`

12. **MixAndMatchCalibrator**
    - **`parametric_calibrator`**: Type of parametric calibrator used. Choose from:
      - `'TemperatureScalingCalibrator'`
      - `'PlattCalibrator'`
      - `'VectorScalingCalibrator'`
      - `'MatrixScalingCalibrator'`
      - `'BetaCalibrator'`
      - `'MetaCalibrator'`
      - `'DirichletCalibrator'`
      - `'AdaptiveTemperatureScalingCalibrator'`
      - Default: `'TemperatureScalingCalibrator'`
    - **`nonparametric_calibrator`**: Type of nonparametric calibrator used. Choose from:
      - `'IsotonicCalibrator'`
      - `'EmpiricalBinningCalibrator'`
      - `'HistogramCalibrator'`
      - Default: `'IsotonicCalibrator'`

13. **ProbabilityTreeCalibrator**
    - **`max_depth`**: Maximum depth of the decision tree used for calibration.
      - Default: `5`
    - **`min_samples_leaf`**: Minimum number of samples required to be at a leaf node.
      - Default: `15`

## Usage Steps

Here's a step-by-step guide on how to use the calibration algorithms with a model's predictions:

### 1. Prepare Your Data
Ensure that you have:
- Model predictions as logits or probabilities in a NumPy array (`(n_samples, n_classes)`)
- Ground truth labels as a NumPy array (`(n_samples,)`)

### 2. Initialize a Calibrator
Choose your calibrator and specify hyperparameters:

```
from calibration_algorithms.adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator
from calibration_algorithms.dirichlet import DirichletCalibrator
from calibration_algorithms.probability_tree import ProbabilityTreeCalibrator

# For Dirichlet Calibration
calibrator = DirichletCalibrator(
    lr=0.01,           # learning rate
    max_iter=1000,      # maximum iterations
    seed=42            # random seed
)

# For Adaptive Temperature Scaling
calibrator = AdaptiveTemperatureScalingCalibrator(
    mode='hybrid',      # 'linear', 'entropy', or 'hybrid'
    confidence_bins=10, # number of confidence bins
    entropy_bins=10,    # number of entropy bins
    initial_T=1.0,      # initial temperature value
    lr_tempscaling=0.01,
    max_iter_tempscaling=1000,
    seed=42
)

# For Probability Tree Calibration
calibrator = ProbabilityTreeCalibrator(
    max_depth=5,        # maximum depth of decision tree
    min_samples_leaf=15, # minimum samples per leaf node
    seed=42
)
```

### 3. Fit the Calibrator
Train the calibrator on your validation set:

```
# Fit the calibrator with model predictions and ground truth labels
calibrator.fit(
    predictions=val_predictions,     # NumPy array of shape (n_samples, n_classes)
    ground_truth=val_labels     # NumPy array of shape (n_samples,)
)

# Note: For ProbabilityTreeCalibrator, you also need to provide features:
# calibrator.fit(
#     X=val_features,              # NumPy array of shape (n_samples, n_features)
#     logits=val_predictions,      # NumPy array of shape (n_samples, n_classes)
#     y=val_labels                 # NumPy array of shape (n_samples,)
# )
```

### 4. Generate Calibrated Probabilities
Use the trained calibrator to get calibrated probabilities on new data:

```
# Get calibrated probabilities for test data
calibrated_probs = calibrator.predict(test_predictions)
print("Calibrated Probabilities:", calibrated_probs[:5])  # Print the first 5 samples

# Note: For ProbabilityTreeCalibrator, you also need to provide features:
# calibrated_probs = calibrator.predict(
#     X=test_features,             # NumPy array of shape (n_samples, n_features)
#     logits=test_predictions      # NumPy array of shape (n_samples, n_classes)
# )
```