import numpy as np

from smartcal.calibration_algorithms import HistogramCalibrator


# Configuration
num_samples = 1000
num_classes_multi = 3
num_bins = 10
seed = 42

np.random.seed(seed)

# Softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Generate synthetic training data
logits_multi = np.random.randn(num_samples, num_classes_multi)
probabilities_multi = softmax(logits_multi)
ground_truth_multi = np.random.randint(0, num_classes_multi, size=num_samples)

# Generate test logits and softmax probabilities
test_logits = np.random.randn(10, num_classes_multi)
test_probabilities = softmax(test_logits)

# Try each histogram calibrator type
histogram_types = ["HISTOGRAM", "HISTOGRAMMARGINAL"]

for hist_type in histogram_types:
    print(f"\n--- Running HistogramCalibrator: {hist_type} ---")

    calibrator = HistogramCalibrator(calibrator_type=hist_type, num_bins=num_bins)
    calibrator.fit(probabilities_multi, ground_truth_multi)
    calibrated_probs = calibrator.predict(test_probabilities)

    print("\nBefore calibration (softmax):")
    print(np.round(test_probabilities[:5], 4))

    print("\nAfter calibration:")
    print(np.round(calibrated_probs[:5], 4))

    print("\nCalibrator meta data:")
    print(calibrator.metadata)