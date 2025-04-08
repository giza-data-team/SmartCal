from SmartCal.calibration_algorithms.empirical_binning import EmpiricalBinningCalibrator
from SmartCal.calibration_algorithms.isotonic import IsotonicCalibrator
from SmartCal.calibration_algorithms.beta import BetaCalibrator
from SmartCal.calibration_algorithms.vector_scaling import VectorScalingCalibrator
from SmartCal.calibration_algorithms.temperature_scaling import TemperatureScalingCalibrator
from SmartCal.calibration_algorithms.dirichlet import DirichletCalibrator
from SmartCal.calibration_algorithms.meta import MetaCalibrator
from SmartCal.calibration_algorithms.matrix_scaling import MatrixScalingCalibrator
from SmartCal.calibration_algorithms.platt.platt import PlattCalibrator
from SmartCal.calibration_algorithms.histogram.histogram import HistogramCalibrator
from SmartCal.calibration_algorithms.mix_and_match import MixAndMatchCalibrator
from SmartCal.calibration_algorithms.adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator

from enum import Enum


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
