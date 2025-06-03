import pandas as pd
import os

def load_data():
    """Load and tag baseline and meta model data."""
    baseline_df = pd.read_csv("experiments/meta_model_eval/Results/baseline_results.csv")
    meta_model_df = pd.read_csv("experiments/meta_model_eval/Results/meta_model_results.csv")

    baseline_df['Source'] = 'Baseline'
    meta_model_df['Source'] = 'Meta Model'

    # Filter rows where Meta_Model_Metric matches Evaluation_Metric
    meta_model_df = meta_model_df[
        meta_model_df['Meta_Model_Metric'].str.lower() == meta_model_df['Evaluation_Metric'].str.lower()
        ]

    return baseline_df, meta_model_df


def clean_merge_dataframes(baseline_df, meta_model_df):
    """
    Merge the two dataframes while properly handling common and unique columns.
    """
    common_cols = ["Dataset_Name", "Model_Type", "N", "Evaluation_Metric", "Problem_Type", "Dataset_Type"]

    baseline_cols = [
        "Selected_Calibrators",
        "Best_Calibrator", "Best_Performance_Value"
    ]

    meta_model_cols = [
        "Predicted_Calibrators", "Best_Calibrator",
        "Best_Performance_Value", "Best_Possible_Performance",
        "Meta_Model_Confidence", "Meta_Model_Version", "Is_Best_Calibrator",
        "Meta_Model_Metric"
    ]

    baseline_renamed = baseline_df[common_cols + baseline_cols].copy()
    meta_model_renamed = meta_model_df[common_cols + meta_model_cols].copy()

    # Rename conflicting columns
    conflicting_cols = set(baseline_cols) & set(meta_model_cols)
    for col in conflicting_cols:
        if col in baseline_renamed.columns:
            baseline_renamed.rename(columns={col: f"Baseline_{col}"}, inplace=True)
        if col in meta_model_renamed.columns:
            meta_model_renamed.rename(columns={col: f"MetaModel_{col}"}, inplace=True)

    # Merge dataframes
    merged_df = pd.merge(
        baseline_renamed,
        meta_model_renamed,
        on=common_cols,
        how='outer'
    )

    return merged_df


def determine_winner(row):
    """
    Determine winner based on calibrator selection and performance comparison.
    """
    meta_model_calibrator = row['MetaModel_Best_Calibrator']
    baseline_calibrator = row['Baseline_Best_Calibrator']
    meta_model_performance = row['MetaModel_Best_Performance_Value']
    baseline_performance = row['Baseline_Best_Performance_Value']

    if meta_model_calibrator == baseline_calibrator:
        return "tie"
    elif meta_model_performance < baseline_performance:
        return "Meta Model"
    elif meta_model_performance > baseline_performance:
        return "Random Selection"
    else:
        return "tie"


def main():
    # Load and clean data
    baseline_df, meta_model_df = load_data()
    merged_df = clean_merge_dataframes(baseline_df, meta_model_df)

    # Rename and apply winner determination logic 
    merged_df = merged_df.rename(columns={
        'Selected_Calibrators': 'Baseline_Selected_Calibrators',
        'N': 'K'
    })

    merged_df['Winner'] = merged_df.apply(determine_winner, axis=1)

    # Create Results directory if it doesn't exist
    results_dir = "analysis/meta_model_eval/Results"
    os.makedirs(results_dir, exist_ok=True)

    # Save to CSV
    merged_df.to_csv(f"{results_dir}/combined_results.csv", index=False)
    print("Merged results saved to combined_results.csv")


if __name__ == "__main__":
    main()
