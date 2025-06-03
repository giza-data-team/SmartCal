import numpy as np
import torch
from torch.nn.functional import softmax as torch_softmax

from smartcal.calibration_algorithms import TemperatureScalingCalibrator

# Simulate synthetic multi-class logits and labels
np.random.seed(42)
torch.manual_seed(42)

n_samples = 100
n_classes = 3
logits = np.random.randn(n_samples, n_classes) * 3
labels = np.random.choice(n_classes, size=n_samples)

# Uncalibrated softmax probabilities
uncalibrated_probs = torch_softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()

# Initialize calibrator
calibrator = TemperatureScalingCalibrator(
    initial_T=1.0,
    lr_tempscaling=0.01,
    max_iter_tempscaling=100
)

# Fit and predict
calibrator.fit(logits, labels)
calibrated_probs = calibrator.predict(logits)

# Output
print("\n--- Temperature Scaling Calibration ---")
print("\nBefore calibration (softmax):")
print(np.round(uncalibrated_probs[:5], 4))

print("\nAfter calibration (temperature scaling):")
print(np.round(calibrated_probs[:5], 4))

# Display optimized temperature
print("\nOptimized temperature T:")
print(calibrator.metadata["params"]["optimized_temperature"])

print("\nCalibrator meta data:")
print(calibrator.metadata)
