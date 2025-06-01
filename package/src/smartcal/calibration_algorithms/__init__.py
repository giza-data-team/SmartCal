__all__ = ["HistogramCalibrator", "PlattCalibrator", "AdaptiveTemperatureScalingCalibrator", "TemperatureScalingCalibrator", 
        "IsotonicCalibrator", "MetaCalibrator", "BetaCalibrator", "DirichletCalibrator", "MatrixScalingCalibrator", "VectorScalingCalibrator", 
        "EmpiricalBinningCalibrator", "MixAndMatchCalibrator", "ProbabilityTreeCalibrator"]

def _lazy_imports():
    global HistogramCalibrator, PlattCalibrator, AdaptiveTemperatureScalingCalibrator, TemperatureScalingCalibrator, \
        IsotonicCalibrator, MetaCalibrator, BetaCalibrator, DirichletCalibrator, MatrixScalingCalibrator, VectorScalingCalibrator, \
        EmpiricalBinningCalibrator, MixAndMatchCalibrator, ProbabilityTreeCalibrator
    
    from smartcal.calibration_algorithms.histogram.histogram import HistogramCalibrator
    from smartcal.calibration_algorithms.platt.platt import PlattCalibrator
    from smartcal.calibration_algorithms.adaptive_temperature_scaling import AdaptiveTemperatureScalingCalibrator
    from smartcal.calibration_algorithms.temperature_scaling import TemperatureScalingCalibrator
    from smartcal.calibration_algorithms.isotonic import IsotonicCalibrator
    from smartcal.calibration_algorithms.meta import MetaCalibrator
    from smartcal.calibration_algorithms.beta import BetaCalibrator
    from smartcal.calibration_algorithms.dirichlet import DirichletCalibrator
    from smartcal.calibration_algorithms.matrix_scaling import MatrixScalingCalibrator
    from smartcal.calibration_algorithms.vector_scaling import VectorScalingCalibrator
    from smartcal.calibration_algorithms.empirical_binning import EmpiricalBinningCalibrator
    from smartcal.calibration_algorithms.mix_and_match import MixAndMatchCalibrator
    from smartcal.calibration_algorithms.probability_tree import ProbabilityTreeCalibrator

_lazy_imports()
