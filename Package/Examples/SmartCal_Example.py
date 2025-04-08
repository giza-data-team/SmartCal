import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split

# Add the parent directory to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SmartCal framework
from Package.src.SmartCal.SmartCal.SmartCal import SmartCal


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute row-wise softmax for a batch of logits."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def generate_sample_data(n_samples: int = 200, n_classes: int = 3, seed: int = 42):
    """Generate synthetic logits and true labels for demo purposes."""
    np.random.seed(seed)
    logits = np.random.randn(n_samples, n_classes) * 2
    probabilities = softmax(logits)
    y_true = np.random.choice(n_classes, size=n_samples)
    return y_true, probabilities


def main():
    try:
        print("[Step 1] Generating synthetic prediction data...")
        y_true, probs = generate_sample_data()

        print("[Step 2] Splitting into calibration and test sets (70/30)...")
        y_cal, y_test, p_cal, p_test = train_test_split(
            y_true, probs, test_size=0.3, random_state=42, stratify=y_true
        )
        print(f"Calibration samples: {len(y_cal)} | Test samples: {len(y_test)}")

        print("[Step 3] Initializing SmartCal and recommending calibrators...")
        smartcal = SmartCal()
        recommendations = smartcal.recommend_calibrators(y_cal, p_cal, n=5, metric='ECE')
        for name, score in recommendations:
            print(f" - {name}: ECE = {score:.3f}")

        print("[Step 4] Fitting best calibrator on calibration set...")
        best_calibrator = smartcal.best_fitted_calibrator(y_cal, p_cal, n_iter=10, metric='ECE')
        print(f"Best Calibrator Selected: {type(best_calibrator).__name__}")

        print("[Step 5] Applying calibration to test set...")
        p_test_calibrated = best_calibrator.predict(p_test)

        print("\n[Step 6] Evaluating calibration results on test set:")
        np.set_printoptions(precision=3, suppress=True)
        print("\nOriginal Test Probabilities (first 5):")
        print(p_test[:5])

        print("\nCalibrated Test Probabilities (first 5):")
        print(p_test_calibrated[:5])

    except Exception as e:
        print(f"\n[ERROR] Calibration process failed: {e}")


if __name__ == "__main__":
    main()
