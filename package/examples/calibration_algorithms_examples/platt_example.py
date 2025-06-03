import numpy as np

from smartcal.calibration_algorithms import PlattCalibrator


# Settings
num_samples = 100
num_bins = 10
num_classes_multi = 3
seed = 42
np.random.seed(seed)

# Generate logits and labels
multi_logits = np.random.randn(num_samples, num_classes_multi) * 3
multi_ground_truth = np.random.randint(0, num_classes_multi, size=num_samples)

# Softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

uncalibrated_probs = softmax(multi_logits)

# Try each Platt calibrator type
platt_types = ["PLATT", "PLATTBINNER", "PLATTBINNERMARGINAL"]

for platt_type in platt_types:
    print(f"\n--- Testing PlattCalibrator: {platt_type} ---")

    # Multi-class case
    calibrator_multi = PlattCalibrator(calibrator_type=platt_type, num_bins=num_bins)
    calibrator_multi.fit(multi_logits, multi_ground_truth)
    calibrated_multi = calibrator_multi.predict(multi_logits)

    print("\nBefore calibration (softmax):")
    print(np.round(uncalibrated_probs[:5], 4))

    print("\nAfter calibration:")
    print(np.round(calibrated_multi[:5], 4))

    print("\nCalibrator meta data:")
    print(calibrator_multi.metadata)
