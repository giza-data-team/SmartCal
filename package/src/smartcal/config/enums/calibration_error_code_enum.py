from enum import IntEnum

class CalibrationErrorCode(IntEnum):
    NANS_DETECTED = 1001
    ALMOST_CONSTANT_PROBS = 1002
    # You can add more codes here as needed
