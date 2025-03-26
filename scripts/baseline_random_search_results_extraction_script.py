import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baselines.baseline_random_search import get_all_calibrator_combinations, get_n_random_calibration_combinations, apply_calibration_with_cv, process_random_baseline
from experiment_manager.db_connection import SessionLocal
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from experiment_manager.models import BenchmarkingExperiment_V2

# Example usage:
if __name__ == "__main__":
    #combinations = get_all_calibrator_combinations()
    #random_combinations = get_n_random_calibration_combinations(combinations, 50)
    #print(random_combinations)

    # Simulated uncalibrated probabilities and labels
    '''
    np.random.seed(42)
    uncalibrated_probs_valid = np.random.rand(100, 2)  # 100 samples, 2 class probabilities
    y_valid = np.random.randint(0, 2, 100)  # Binary labels

    uncalibrated_probs_test = np.random.rand(50, 2)  # 50 test samples
    y_test = np.random.randint(0, 2, 50)  # Binary labels

    # Example usage:
    example_hyperparameters = {
        "Calibration_Algorithm": "TEMPERATURESCALING",
        "initial_T": 1.0,
        "lr_tempscaling": 0.01,
        "max_iter_tempscaling": 500
    }

    # Apply calibration
    calibrated_probs_test, calibrated_probs_valid = apply_calibration_with_cv(example_hyperparameters, uncalibrated_probs_valid, y_valid, uncalibrated_probs_test)
    '''
    with SessionLocal() as db:
        process_random_baseline(db, BenchmarkingExperiment_V2, [10], "Results/baseline_random_search_results.csv")