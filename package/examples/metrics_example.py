import numpy as np

from smartcal.metrics import ECE
from smartcal.metrics import MCE
from smartcal.metrics import ConfECE
from smartcal.metrics import calculate_brier_score
from smartcal.metrics import calculate_calibration_curve


# Define parameters
num_bins = 10
confidence_threshold = 0.5

# Multiclass input (ground truth and predicted probabilities)
y_true = np.array([0, 1, 2, 1, 0])
y_prob = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.7, 0.2],
    [0.2, 0.2, 0.6],
    [0.3, 0.6, 0.1],
    [0.6, 0.2, 0.2]
])

# Get predicted labels (argmax of predicted probabilities)
y_pred = np.argmax(y_prob, axis=1)

# ECE
ece = ECE(num_bins=num_bins)
ece_value = ece.compute(y_prob, y_pred, y_true)
print("\nECE:")
print("Value:", ece_value)
print("Log:", ece.logger())

# MCE
mce = MCE(num_bins=num_bins)
mce_value = mce.compute(y_prob, y_pred, y_true)
print("\nMCE:")
print("Value:", mce_value)
print("Log:", mce.logger())

# ConfECE
conf_ece = ConfECE(num_bins=num_bins, confidence_threshold=confidence_threshold)
conf_ece_value = conf_ece.compute(y_prob, y_pred, y_true)
print("\nConfECE:")
print("Value:", conf_ece_value)
print("Log:", conf_ece.logger())

# Brier Score
brier_score = calculate_brier_score(y_true, y_prob)
print("\nMulticlass Brier Score:", brier_score)

# Calibration Curve
mean_probs, true_probs, bin_counts = calculate_calibration_curve(
    y_true, y_prob, n_bins=num_bins
)
print("\nCalibration Curve:")
print("Mean Probabilities per Bin:", mean_probs)
print("True Probabilities per Bin:", true_probs)
print("Samples per Bin:", bin_counts)
