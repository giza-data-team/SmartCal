from calibration import PlattBinnerMarginalCalibrator
from smartcal.calibration_algorithms.platt.platt_scaling import PlattScalingCalibrator
from smartcal.calibration_algorithms.platt.platt_scaling_binning import PlattBinnerScalingCalibrator

from enum import Enum


class PlattTypesEnum(Enum):
    PLATT = PlattScalingCalibrator
    PLATTBINNER = PlattBinnerScalingCalibrator
    PLATTBINNERMARGINAL = PlattBinnerMarginalCalibrator

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)
