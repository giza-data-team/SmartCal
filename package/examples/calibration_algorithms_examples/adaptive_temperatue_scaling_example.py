import numpy as np
import torch
from torch.nn.functional import softmax as torch_softmax

from smartcal.calibration_algorithms import AdaptiveTemperatureScalingCalibrator


# Simulate synthetic multi-class logits and labels (it works for both binary and multi-class data)
np.random.seed(42)
torch.manual_seed(42)

n_samples = 100
n_classes = 3
logits = np.random.randn(n_samples, n_classes) * 3
labels = np.random.choice(n_classes, size=n_samples)

# Uncalibrated softmax probabilities
uncalibrated_probs = torch_softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()

# Available modes: "linear", "entropy", "hybrid", default: "hybrid"
mode = "hybrid"  # Try switching to "linear" or "entropy" for experimentation

# Note:
# - If mode == 'linear', you must set `confidence_bins` (number of confidence intervals).
# - If mode == 'entropy', you must set `entropy_bins` (number of entropy intervals).
# - If mode == 'hybrid', you must set both `confidence_bins` and `entropy_bins`.
# Each mode initializes a temperature tensor accordingly.

# Initialize calibrator
calibrator = AdaptiveTemperatureScalingCalibrator(
    mode=mode,
    confidence_bins=5,
    entropy_bins=5,
    initial_T=1.5,
    lr_tempscaling=0.01,
    max_iter_tempscaling=100
)

# Fit and predict
calibrator.fit(logits, labels)
calibrated_probs = calibrator.predict(logits)

# Output comparison
print(f"\n--- Adaptive Temperature Scaling (mode: {mode}) ---")
print("\nBefore calibration (softmax):")
print(np.round(uncalibrated_probs[:5], 4))

print("\nAfter calibration (ATS):")
print(np.round(calibrated_probs[:5], 4))

print("\nCalibrator meta data:")
print(calibrator.metadata)
