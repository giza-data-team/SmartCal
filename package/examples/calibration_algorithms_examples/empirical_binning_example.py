import numpy as np

from smartcal.calibration_algorithms import EmpiricalBinningCalibrator


# Simulate synthetic 2-class probability predictions and labels
np.random.seed(42)
n_samples = 100
raw_probs = np.random.beta(2, 5, n_samples)  # Simulate overconfident predictions
predictions = np.column_stack([1 - raw_probs, raw_probs])
labels = np.random.binomial(1, 0.5, size=n_samples)

# Uncalibrated predictions
print("\n--- Empirical Binning Calibration ---")
print("\nBefore calibration:")
print(np.round(predictions[:5], 4))

# Initialize calibrator
calibrator = EmpiricalBinningCalibrator(n_bins=10)

# Fit and predict
calibrator.fit(predictions, labels)
calibrated_probs = calibrator.predict(predictions)

# Output
print("\nAfter calibration (empirical binning):")
print(np.round(calibrated_probs[:5], 4))

print("\nBin Boundaries:")
print(np.round(calibrator.bin_boundaries, 4))

print("\nBin Probabilities:")
print(np.round(calibrator.bin_probabilities, 4))

print("\nCalibrator metadata:")
print({
    "n_bins": calibrator.n_bins,
    "fitted": calibrator.fitted
})
