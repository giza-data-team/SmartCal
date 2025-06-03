import numpy as np
from scipy.special import softmax

from smartcal.calibration_algorithms import BetaCalibrator


# Simulate synthetic multi-class softmax probabilities and labels
np.random.seed(42)
n_samples = 100
n_classes = 3

# Generate logits and apply softmax
logits = np.random.randn(n_samples, n_classes) * 3
probabilities = softmax(logits, axis=1)

# Simulate ground truth labels (0 to n_classes-1)
labels = np.random.choice(n_classes, size=n_samples)

# Initialize and fit the BetaCalibrator
calibrator = BetaCalibrator(model_type="abm") # mode can be: "abm", "am", or "ab", default: "abm"
calibrator.fit(probabilities, labels)
calibrated_probs = calibrator.predict(probabilities)

# Output
print("\n--- Beta Calibration ---")
print("\nBefore calibration (softmax):")
print(np.round(probabilities[:5], 4))

print("\nAfter calibration (beta):")
print(np.round(calibrated_probs[:5], 4))

print("\nCalibrator meta data:")
print(calibrator.metadata)