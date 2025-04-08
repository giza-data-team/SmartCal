import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from experiment_manager.csv_experiment_manager import CSVExperimentManager
from Package.src.SmartCal.config.enums.experiment_type_enum import ExperimentType
from Package.src.SmartCal.config.enums.dataset_types_enum import DatasetTypesEnum

# Configure logging
logging.basicConfig(level=logging.INFO)

# intilize an experiment manager object with required input
# [TABULAR, IMAGE, LANGUAGE], [BENCHMARKING_V2, KNOWLEDGE_BASE_V2]

# List of seeds to use
# split_seeds = [42, 123, 456, 789, 101]
split_seeds = [26, 30, 42, 78, 101]


for trial_number, seed in enumerate(split_seeds, 1):
    # Create a new manager for each seed
    exp_manager = CSVExperimentManager(
        DatasetTypesEnum.TABULAR,
        ExperimentType.BENCHMARKING_V2,
        split_seed=seed,
        use_kfold=True,
        trial_number=trial_number
    )

    logging.info(f"Running experiments with seed {seed}, trail number {trial_number}")

    # Run all experiments for this seed
    exp_manager.run_all_experiments(add_failed=True, use_default_hyperparams=True)

    logging.info(f"Completed experiments with seed {seed}, trail number {trial_number}")

logging.info("All seeds completed")

