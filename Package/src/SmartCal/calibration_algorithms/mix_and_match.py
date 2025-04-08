import numpy as np
from .calibration_base import CalibratorBase
from SmartCal.utils.timer import time_operation
from .temperature_scaling import TemperatureScalingCalibrator
from .isotonic import IsotonicCalibrator
from .beta import BetaCalibrator
from .dirichlet import DirichletCalibrator
from .empirical_binning import EmpiricalBinningCalibrator
from .matrix_scaling import MatrixScalingCalibrator
from .platt.platt_scaling import PlattScalingCalibrator
from .vector_scaling import VectorScalingCalibrator
from .histogram.histogram import HistogramCalibrator
from .meta import MetaCalibrator
from .adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator

class MixAndMatchCalibrator(CalibratorBase):
    """
    Combines parametric and non-parametric calibration methods.
    First applies a parametric calibration
    then refines with a non-parametric calibration
    -------
    parametric_calibrator :
        Must be one of the following parametric calibrators:
        - TemperatureScaling (default)
        - PlattScalingCalibrator
        - VectorScalingCalibrator
        - MatrixScaling
        - BetaCalibrator
        - Dirichlet

    nonparametric_calibrator :
        Must be one of the following non-parametric calibrators:
        - IsotonicCalibrator (default)
        - EmpiricalBinningCalibrator
        - HistogramCalibrator

    """

    def __init__(self, parametric_calibrator = "TemperatureScalingCalibrator",nonparametric_calibrator =  "IsotonicCalibrator", seed=CalibratorBase.DEFAULT_SEED):
        super().__init__(seed)
        self.timing = {}


        parametric_mapping = {
            "TemperatureScalingCalibrator": TemperatureScalingCalibrator,
            "PlattScalingCalibrator": PlattScalingCalibrator,
            "VectorScalingCalibrator": VectorScalingCalibrator,
            "MatrixScalingCalibrator": MatrixScalingCalibrator,
            "BetaCalibrator": BetaCalibrator,
            "MetaCalibrator": MetaCalibrator,
            "DirichletCalibrator": DirichletCalibrator,
            "AdaptiveTemperatureScalingCalibrator": AdaptiveTemperatureScalingCalibrator
        }

        nonparametric_mapping = {
            "IsotonicCalibrator": IsotonicCalibrator,
            "EmpiricalBinningCalibrator": EmpiricalBinningCalibrator,
            "HistogramCalibrator": HistogramCalibrator
        }


        if not parametric_calibrator or not nonparametric_calibrator:
            raise ValueError("Exactly one parametric and one non-parametric calibrator must be provided.")

        self.parametric_calibrator = parametric_mapping[parametric_calibrator]()
        self.nonparametric_calibrator = nonparametric_mapping[nonparametric_calibrator]()
        self.fitted = False

        self.metadata["params"].update({
            "parametric_calibrator_type": self.parametric_calibrator.__class__.__name__,
            "nonparametric_calibrator_type": self.nonparametric_calibrator.__class__.__name__,
        })

    @time_operation
    def fit(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Fits the parametric calibrator first, then the non-parametric calibrator.

        Args:
            predictions (np.ndarray): Model logits or probabilities of shape (n_samples, n_classes).
            ground_truth (np.ndarray): True labels of shape (n_samples,).
        """
        self.set_random_seed(self.seed)

        predictions, ground_truth = self.validate_inputs(predictions, ground_truth)

        # Fit the parametric calibrator
        self.parametric_calibrator.fit(predictions, ground_truth)
        parametric_calibrated = self.parametric_calibrator.predict(predictions)

        # Fit the non-parametric calibrator on the parametric calibrated probabilities
        self.nonparametric_calibrator.fit(parametric_calibrated, ground_truth)
        self.fitted = True

        self.metadata["dataset_info"].update({
            "n_samples": predictions.shape[0],
            "n_classes": predictions.shape[1],
        })

    @time_operation
    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Generates calibrated probabilities using the mix-and-match approach.

        Args:
            test_data (np.ndarray): Uncalibrated model outputs of shape (n_samples, n_classes).

        Returns:
            np.ndarray: Calibrated probabilities of shape (n_samples, n_classes).
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # Apply parametric calibration
        parametric_calibrated = self.parametric_calibrator.predict(test_data)

        # Apply non-parametric calibration
        return self.nonparametric_calibrator.predict(parametric_calibrated)

    def get_timing(self):
        return self.timing
