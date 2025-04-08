import logging
logging.basicConfig(level=logging.WARNING)  # This line configures the logging module to display only warnings or more critical messages in the terminal.

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from baseline_random_search.baseline_random_search import process_random_baseline
from experiment_manager.db_connection import SessionLocal
from experiment_manager.models import BenchmarkingExperiment_V2

if __name__ == "__main__":
    with SessionLocal() as db:
        process_random_baseline(db, BenchmarkingExperiment_V2, "experiments/end_to_end_eval/Results/baseline_random_search_results.csv")