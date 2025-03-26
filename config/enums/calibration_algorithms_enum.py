from calibration_algorithms.empirical_binning import EmpiricalBinningCalibrator
from calibration_algorithms.isotonic import IsotonicCalibrator
from calibration_algorithms.beta import BetaCalibrator
from calibration_algorithms.imax import ImaxCalibrator
from calibration_algorithms.vector_scaling import VectorScalingCalibrator
from calibration_algorithms.temperature_scaling import TemperatureScalingCalibrator
from calibration_algorithms.dirichlet import DirichletCalibrator
from calibration_algorithms.meta import MetaCalibrator
from calibration_algorithms.matrix_scaling import MatrixScalingCalibrator
from calibration_algorithms.platt.platt import PlattCalibrator
from calibration_algorithms.histogram.histogram import HistogramCalibrator
from calibration_algorithms.mix_and_match import MixAndMatchCalibrator
from calibration_algorithms.adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator

from enum import Enum


class CalibrationAlgorithmTypesEnum(Enum):
    EMPIRICALBINNING = EmpiricalBinningCalibrator  
    ISOTONIC = IsotonicCalibrator 
    BETA = BetaCalibrator  
    IMAX = ImaxCalibrator
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
