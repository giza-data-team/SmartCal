import numpy as np
import torch
from torch.nn.functional import softmax as torch_softmax
from smartcal.calibration_algorithms import DirichletCalibrator


# Simulate synthetic logits and labels
np.random.seed(42)
torch.manual_seed(42)

n_samples = 100
n_classes = 3

logits = np.random.randn(n_samples, n_classes) * 3
labels = np.random.choice(n_classes, size=n_samples)

# Uncalibrated softmax probabilities
uncalibrated_probs = torch_softmax(torch.tensor(logits, dtype=torch.float32), dim=1).numpy()

# Initialize and fit the DirichletCalibrator
calibrator = DirichletCalibrator(lr=0.01, max_iter=100)
calibrator.fit(logits, labels)
calibrated_probs = calibrator.predict(logits)

# Output
print("\n--- Dirichlet Calibration ---")
print("\nBefore calibration (softmax):")
print(np.round(uncalibrated_probs[:5], 4))

print("\nAfter calibration (Dirichlet):")
print(np.round(calibrated_probs[:5], 4))

print("\nCalibrator meta data:")
print(calibrator.metadata)