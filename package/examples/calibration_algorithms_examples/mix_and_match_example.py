import numpy as np
from scipy.special import softmax
from smartcal.calibration_algorithms import MixAndMatchCalibrator

# Simulate synthetic logits and labels
np.random.seed(42)
n_samples = 100
n_classes = 3
logits = np.random.randn(n_samples, n_classes) * 3
labels = np.random.choice(n_classes, size=n_samples)

# You can use softmax(logits) if calibrators expect probabilities.
# But since most of your parametric ones expect logits, we’ll use raw logits directly.
uncalibrated_probs = softmax(logits, axis=1)

# Try different combinations (e.g., BetaCalibrator + IsotonicCalibrator)
parametric = "TemperatureScalingCalibrator"
nonparametric = "IsotonicCalibrator"

# Initialize and fit MixAndMatchCalibrator
calibrator = MixAndMatchCalibrator(parametric_calibrator=parametric, nonparametric_calibrator=nonparametric)
calibrator.fit(logits, labels)
calibrated_probs = calibrator.predict(logits)

# Output results
print(f"\n--- MixAndMatch Calibration ---")
print(f"\nParametric: {parametric}, Non-parametric: {nonparametric}")

print("\nBefore calibration (softmax):")
print(np.round(uncalibrated_probs[:5], 4))

print("\nAfter calibration (MixAndMatch):")
print(np.round(calibrated_probs[:5], 4))

print("\nCalibrator meta data:")
print(calibrator.metadata)
