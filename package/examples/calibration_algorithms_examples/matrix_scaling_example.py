import numpy as np
import torch
from torch.nn.functional import softmax

from smartcal.calibration_algorithms import MatrixScalingCalibrator


# Input logits and labels
logits = np.array([
    [2.5, 1.0, -1.2],
    [1.2, 3.5, 0.5],
    [0.5, -0.5, 3.0]
])
labels = np.array([0, 1, 2])

# Uncalibrated softmax probabilities
uncalibrated_probs = softmax(torch.tensor(logits), dim=1).numpy()

# Initialize calibrator
calibrator = MatrixScalingCalibrator(lr=0.01, max_iter=100)

# Fit and predict
calibrator.fit(logits, labels)
calibrated_probs = calibrator.predict(logits)

# Output
print("\n--- Matrix Scaling Calibration ---")
print("\nBefore calibration (softmax):")
print(np.round(uncalibrated_probs, 4))

print("\nAfter calibration (matrix scaling):")
print(np.round(calibrated_probs, 4))

print("\nCalibrator meta data:")
print(calibrator.metadata)
