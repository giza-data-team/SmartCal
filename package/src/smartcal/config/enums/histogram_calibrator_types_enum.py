from calibration import HistogramMarginalCalibrator
from smartcal.calibration_algorithms.histogram.histogram_scaling import HistogramScalingCalibrator

from enum import Enum


class HistogramTypesEnum(Enum):
    HISTOGRAM = HistogramScalingCalibrator
    HISTOGRAMMARGINAL = HistogramMarginalCalibrator

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
