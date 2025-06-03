import numpy as np
import torch
from torch.nn.functional import softmax

from smartcal.calibration_algorithms import VectorScalingCalibrator


# Input logits and labels
logits = np.array([
    [2.5, 1.0, -1.2],
    [1.2, 3.5, 0.5],
    [0.5, -0.5, 3.0]
])
labels = np.array([0, 1, 2])

# Initialize calibrator
calibrator = VectorScalingCalibrator(lr=0.01, max_iter=100)

# Get uncalibrated probabilities
uncalibrated_probs = softmax(torch.tensor(logits), dim=1).numpy()

# Fit and predict
calibrator.fit(logits, labels)
calibrated_probs = calibrator.predict(logits)

# Output
print("\n--- Vector Scaling Calibration ---")
print("Before calibration (softmax):")
print(np.round(uncalibrated_probs, 4))

print("\nAfter calibration (vector scaling):")
print(np.round(calibrated_probs, 4))

print("\n Calibrator meta data:")
print(calibrator.metadata)
