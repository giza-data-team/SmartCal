import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
import pandas as pd
import numpy as np
import time

from Package.src.SmartCal.config.configuration_manager import ConfigurationManager

# Load configuration manager
config_manager = ConfigurationManager()


def save_results_to_csv(df, original_filename):
    """
    Saves the results to a CSV file in the Results folder
    with the original filename + _summary
    """
    # Extract just the filename from the path
    results_dir = config_manager.results_dir
    if not isinstance(results_dir, str) or not results_dir:
        raise ValueError("Invalid results directory from config_manager")

    print(f"Saving to directory: {results_dir}")  # Debugging output

    # Extract the base filename
    base_filename = os.path.basename(original_filename)

    # Modify the filename
    output_filename = base_filename.replace('.csv', '_summary.csv')

    # Construct the full path
    output_path = os.path.join(results_dir, output_filename)

    # Save the DataFrame
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    return output_filename


def aggregate_metrics(filename):
    """
    Aggregate metrics across trials, keeping only numeric data with mean and std
    """
    # Read the CSV file
    print("Reading CSV file...")
    df = pd.read_csv(filename)

    # Start timing
    start_time = time.time()
    print("Starting metrics aggregation...")

    # Print basic dataset info
    print(f"\nDataset Overview:")
    print(f"Total number of rows: {len(df)}")
    print(f"Unique datasets: {df['dataset_name'].nunique()}")
    print(f"Unique classification models: {df['classification_model'].nunique()}")
    print(f"Unique calibration algorithms: {df['calibration_algorithm'].nunique()}")

    # Columns to preserve as is (no mean/std calculation)
    preserve_columns = [
        'split_ratio_test', 'split_ratio_cal', 'split_ratio_train',
        'n_instances_cal_set', 'problem_type', 'trial_no', 'split_seed', "ground_truth_cal_set", "ground_truth_test_set"
    ]

    # Dynamically identify numeric columns
    numeric_only = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove preserved columns from numeric columns
    numeric_columns = [
        col for col in numeric_only
        if col not in preserve_columns and col != 'trial_no' and col != 'split_seed'
    ]

    # Group by these columns to maintain experimental context
    group_columns = [
        'dataset_name',
        'classification_model',
        'calibration_algorithm'
    ]

    # Initialize results dictionary
    results_list = []

    # Group and aggregate
    groups = list(df.groupby(group_columns))
    print(f"\nTotal unique groups to process: {len(groups)}")

    for i, (group_key, group) in enumerate(groups, 1):
        # Progress tracking
        print(f"\nProcessing group {i}/{len(groups)}: {group_key}")
        print(f"Trials in this group: {len(group)}")

        # Create new result row
        result = {}

        # Add group key columns
        for j, key in enumerate(group_columns):
            result[key] = group_key[j]

        # Add preserved columns
        for col in preserve_columns[0:5]:
            if col in group.columns:
                result[col] = group[col].iloc[0]

        # Process numeric columns with mean and std
        print("Processing numeric columns...")
        for col in numeric_columns:
            result[f'{col}_mean'] = group[col].mean()
            result[f'{col}_std'] = group[col].std()

        # Add number of trials and used seeds
        result['trial_count'] = len(group)
        result['split_seeds'] = list(group['split_seed'])

        # Add to results list
        results_list.append(result)



    # Convert to DataFrame
    results_df = pd.DataFrame(results_list)

    # Final timing and summary
    end_time = time.time()
    print(f"\nAggregation Complete!")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
    print(f"Number of aggregated groups: {len(results_df)}")

    return results_df


# File processing
filename = 'experiments/end_to_end_eval/Results/dump_calibrator_results.csv'
# Perform aggregation
print("\nBeginning Metrics Aggregation")
aggregated_results = aggregate_metrics(filename)
save_results_to_csv(aggregated_results, filename)