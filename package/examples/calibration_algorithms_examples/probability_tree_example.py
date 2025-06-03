import numpy as np
from scipy.special import softmax

from smartcal.calibration_algorithms import ProbabilityTreeCalibrator


# Generate synthetic features, logits, and labels
def generate_synthetic_data(n_samples=1000, n_features=5, n_classes=3, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    logits = np.random.randn(n_samples, n_classes) * 3
    labels = np.random.choice(n_classes, n_samples)
    return X, logits, labels

# Data setup
X, logits, labels = generate_synthetic_data()
uncalibrated_probs = softmax(logits, axis=1)

# Initialize and fit the calibrator
calibrator = ProbabilityTreeCalibrator(max_depth=3, min_samples_leaf=10)
calibrator.fit(X, logits, labels)
calibrated_probs = calibrator.predict(X, logits)

# Output
print("\n--- Probability Tree Calibration ---")
print("\nBefore calibration (softmax):")
print(np.round(uncalibrated_probs[:5], 4))

print("\nAfter calibration (tree):")
print(np.round(calibrated_probs[:5], 4))

print("\nCalibrator meta data:")
print(calibrator.metadata)
