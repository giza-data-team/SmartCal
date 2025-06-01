from enum import Enum

from smartcal.calibration_algorithms.histogram.histogram import HistogramCalibrator
from smartcal.calibration_algorithms.platt.platt import PlattCalibrator
from smartcal.calibration_algorithms.temperature_scaling import TemperatureScalingCalibrator
from smartcal.calibration_algorithms.adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator
from smartcal.calibration_algorithms.isotonic import IsotonicCalibrator
from smartcal.calibration_algorithms.meta import MetaCalibrator
from smartcal.calibration_algorithms.beta import BetaCalibrator
from smartcal.calibration_algorithms.dirichlet import DirichletCalibrator
from smartcal.calibration_algorithms.matrix_scaling import MatrixScalingCalibrator
from smartcal.calibration_algorithms.vector_scaling import VectorScalingCalibrator
from smartcal.calibration_algorithms.mix_and_match import MixAndMatchCalibrator
from smartcal.calibration_algorithms.empirical_binning import EmpiricalBinningCalibrator


class CalibrationAlgorithmTypesEnum(Enum):
    EMPIRICALBINNING = EmpiricalBinningCalibrator  
    ISOTONIC = IsotonicCalibrator 
    BETA = BetaCalibrator  
    TEMPERATURESCALING = TemperatureScalingCalibrator
    DIRICHLET = DirichletCalibrator
    META = MetaCalibrator
    MATRIXSCALING = MatrixScalingCalibrator
    VECTORSCALING = VectorScalingCalibrator
    PLATT = PlattCalibrator
    HISTOGRM = HistogramCalibrator
    MIXANDMATCH = MixAndMatchCalibrator
    AdaptiveTemperatureScaling = AdaptiveTemperatureScalingCalibrator

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"
