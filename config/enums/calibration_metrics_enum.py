from metrics import ECE, MCE, ConfECE

from enum import Enum


class CalibrationMetricsEnum(Enum):
    """Enum to map metric names to their corresponding classes."""
    ECE = ECE
    MCE = MCE
    ConfECE = ConfECE

    @classmethod
    def get_metric_class(cls, metric_name: str):
        """Retrieve the metric class by name."""
        try:
            return cls[metric_name].value
        except KeyError:
            raise ValueError(f"Metric '{metric_name}' is not supported.")
