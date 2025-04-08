from enum import Enum


class ExperimentType(Enum):
    """Enum to map experiment types to their corresponding classes."""
    BENCHMARKING = "benchmarking"
    KNOWLEDGE_BASE = "knowledge_base"
    BENCHMARKING_V2 = "benchmarking_v2"
    KNOWLEDGE_BASE_V2 = "knowledge_base_v2"
    
    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"
 