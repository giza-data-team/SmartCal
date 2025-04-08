import logging

# logging.basicConfig(level=logging.WARNING)  # This line configures the logging module to display only warnings or more critical messages in the terminal.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_manager.experiment_manager import ExperimentManager
from Package.src.SmartCal.config.enums.experiment_type_enum import ExperimentType
from Package.src.SmartCal.config.enums.dataset_types_enum import DatasetTypesEnum


# intilize an experiment manager object with required input
# [TABULAR, IMAGE, LANGUAGE], [BENCHMARKING_V2, KNOWLEDGE_BASE_V2]
exp_manager = ExperimentManager(DatasetTypesEnum.TABULAR, ExperimentType.KNOWLEDGE_BASE_V2)

# Run all experiments
next_config = exp_manager.get_next_experiment(add_failed=True)

# get speific experiment_data
#exp_manager.get_experiment_data(1)

# close db connection
exp_manager.close()