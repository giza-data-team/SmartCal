import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiment_manager.db_connection import SessionLocal


with SessionLocal() as db:
    try:
        while True:
            print(".")

    except KeyboardInterrupt:
        print("\nSession closed. Exiting gracefully.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
