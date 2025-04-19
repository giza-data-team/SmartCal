import numpy as np
from scipy.special import softmax

from smartcal.calibration_algorithms import IsotonicCalibrator


# Simulate synthetic probability predictions and labels
np.random.seed(42)
n_samples = 100
n_classes = 3

# Generate synthetic logits and convert to probabilities
logits = np.random.randn(n_samples, n_classes) * 3
uncalibrated_probs = softmax(logits, axis=1)
labels = np.random.choice(n_classes, size=n_samples)

# Initialize and fit the IsotonicCalibrator
calibrator = IsotonicCalibrator()
calibrator.fit(uncalibrated_probs, labels)
calibrated_probs = calibrator.predict(uncalibrated_probs)

# Output
print("\n--- Isotonic Calibration ---")
print("\nBefore calibration (softmax):")
print(np.round(uncalibrated_probs[:5], 4))

print("\nAfter calibration (isotonic):")
print(np.round(calibrated_probs[:5], 4))

print("\nCalibrator meta data:")
print(calibrator.metadata)
