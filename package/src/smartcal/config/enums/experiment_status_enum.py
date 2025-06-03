from enum import Enum


class Experiment_Status_Enum(Enum):
    """ Enum Class to map the Status of Experiments  """
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    INITIALIZED = "initialized"

    def __str__(self):
        """
        Returns the string representation of the Enum value.

        Returns:
            str: The value of the Enum as a string.
        """
        return self.value
