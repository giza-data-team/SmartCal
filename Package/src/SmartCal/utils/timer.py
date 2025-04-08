import time
import functools
import logging

def time_operation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start

        if args and hasattr(args[0], "__class__"):  # Check if the function is a method of a class
            instance = args[0]

            # Check and update execution time in BaseClassifier (if these attributes exist)
            if func.__name__ == "train" and hasattr(instance, "training_time"):
                instance.training_time = round(elapsed, 5)
            elif func.__name__ == "predict" and hasattr(instance, "testing_time_predict"):
                instance.testing_time_predict = round(elapsed, 5)
            elif func.__name__ == "predict_prob" and hasattr(instance, "testing_time_predictprob"):
                instance.testing_time_predictprob = round(elapsed, 5)

            # Check and update execution time in CalibratorBase (if these attributes exist)
            if func.__name__ == "fit" and hasattr(instance, "metadata"):
                instance.metadata["fit_time"] = round(elapsed, 5)
            elif func.__name__ == "predict" and hasattr(instance, "metadata"):
                instance.metadata["predict_time"] = round(elapsed, 5)

            # Logging (if logger exists)
            if hasattr(instance, 'logger') and instance.logs:
                instance.logger.info(f"{instance.__class__.__name__} - {func.__name__} took {elapsed:.6f} seconds")

            # Store the timing in the instance
            if hasattr(instance, 'timing'):
                instance.timing[func.__name__] = elapsed

        else:  # If it's a standalone function, log its execution time
            logging.info(f"{func.__name__} took {elapsed:.6f} seconds")

        return result

    return wrapper
