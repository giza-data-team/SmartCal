import numpy as np
import torch
from scipy.special import softmax

from smartcal.calibration_algorithms import MetaCalibrator


# Function to generate synthetic logits and labels
def generate_synthetic_data(n_samples=1000, n_classes=3, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    logits = np.random.randn(n_samples, n_classes) * 3
    labels = np.random.choice(n_classes, n_samples)
    return logits, labels

# Generate data
logits, labels = generate_synthetic_data()
uncalibrated_probs = softmax(logits, axis=1)

# Define calibrators to try
calibrators = [
    {"type": "ALPHA", "alpha": 0.1, "acc": None},
    {"type": "ACC", "alpha": None, "acc": 0.85},
]

# Run tests
for config in calibrators:
    print(f"\n--- Running MetaCalibrator: {config['type']} Mode ---")

    calibrator = MetaCalibrator(
        calibrator_type=config["type"],
        alpha=config["alpha"],
        acc=config["acc"]
    )

    calibrator.fit(logits, labels)
    calibrated_probs = calibrator.predict(logits)

    print("\nBefore calibration (softmax):")
    print(np.round(uncalibrated_probs[:5], 4))

    print("\nAfter calibration:")
    print(np.round(calibrated_probs[:5], 4))

    print("\nCalibrator meta data:")
    print(calibrator.metadata)
