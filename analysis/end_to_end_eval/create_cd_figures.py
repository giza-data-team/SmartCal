import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
# Use non-interactive backend to prevent display issues
matplotlib.use('Agg')
from critical_difference_utils import compute_CD, graph_ranks

# Step 1: Read data from CSV and organize by method-sample size combinations
def load_data_from_csv_combined(csv_path):
    """
    Load experimental data from CSV file and organize by metric with method-sample size combinations.
    
    Args:
        csv_path (str): Path to the combined_data_for_comparison.csv file
        
    Returns:
        dict: Dictionary with structure {metric: {method_n_combination: [values]}}
    """
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"CSV loaded successfully. Shape: {df.shape}")
    
    # Define the methods and metrics we want to analyze
    metrics = ['ece', 'mce', 'confece', 'log_loss', 'brier_score']
    sample_sizes = [10, 30, 50]
    
    # Define datasets to exclude for log_loss metric
    excluded_datasets_for_log_loss = [
        'BraidFlow', 'GCM', 'Obesity_Classification', 
        'desharnais', 'drug200', 'heart-long-beach', 'preterm'
    ]
    
    print(f"Available columns in CSV: {list(df.columns)}")
    print(f"Unique values in 'n' column: {sorted(df['n'].unique())}")
    print(f"Unique datasets in CSV: {sorted(df['dataset_name'].unique())}")
    
    results = {}
    
    for metric in metrics:
        print(f"\nProcessing {metric.upper()}...")
        
        # Apply filtering for log_loss metric
        if metric == 'log_loss':
            df_filtered = df[~df['dataset_name'].isin(excluded_datasets_for_log_loss)]
            print(f"  Excluding datasets for log_loss: {excluded_datasets_for_log_loss}")
            print(f"  Filtered data shape: {df_filtered.shape} (original: {df.shape})")
        else:
            df_filtered = df
        
        metric_data = {}
        
        # Add RandomSearch variants for different sample sizes
        for n in sample_sizes:
            df_n = df_filtered[df_filtered['n'] == n]
            column_name = f"randomsearch_{metric}"
            if column_name in df_n.columns:
                values = df_n[column_name].dropna().tolist()
                method_name = f"RandomSearch{n}"
                metric_data[method_name] = values
                print(f"  {method_name}: {len(values)} data points")
        
        # Add SmartCal variants for different sample sizes
        for n in sample_sizes:
            df_n = df_filtered[df_filtered['n'] == n]
            column_name = f"smartcal_{metric}"
            if column_name in df_n.columns:
                values = df_n[column_name].dropna().tolist()
                method_name = f"SmartCal{n}"
                metric_data[method_name] = values
                print(f"  {method_name}: {len(values)} data points")
        
        # Add BetaCal (using all data combined)
        column_name = f"beta_{metric}"
        if column_name in df_filtered.columns:
            values = df_filtered[column_name].dropna().tolist()
            metric_data["BetaCal"] = values
            print(f"  BetaCal: {len(values)} data points")
        
        # Add TempScaling (using all data combined)
        column_name = f"temp_scaling_{metric}"
        if column_name in df_filtered.columns:
            values = df_filtered[column_name].dropna().tolist()
            metric_data["TempScaling"] = values
            print(f"  TempScaling: {len(values)} data points")
        
        results[metric] = metric_data
    
    return results

# Step 2: Calculate average ranks for method-sample size combinations
def calculate_average_ranks_combined(metric_data):
    """
    Calculate average ranks for method-sample size combinations.
    
    Args:
        metric_data (dict): Dictionary with method names as keys and performance lists as values
        
    Returns:
        tuple: (average_ranks_list, method_names_list)
    """
    # Get all method names and their data
    methods = list(metric_data.keys())
    print(f"    Methods to compare: {methods}")
    
    # Find the minimum length to ensure all methods have the same number of data points
    data_lengths = [len(data) for data in metric_data.values() if len(data) > 0]
    if not data_lengths:
        print("Warning: No data available for this metric")
        return [], []
    
    min_length = min(data_lengths)
    
    if min_length == 0:
        print("Warning: No data available for this metric")
        return [], []
    
    print(f"    Using {min_length} data points for ranking calculation")
    
    # Create a DataFrame for ranking
    data_for_ranking = {}
    for method, data in metric_data.items():
        if len(data) >= min_length:
            data_for_ranking[method] = data[:min_length]
        else:
            print(f"    Warning: {method} has only {len(data)} data points, skipping")
    
    if not data_for_ranking:
        print("Warning: No valid data for ranking")
        return [], []
    
    print(f"    Final methods for ranking: {list(data_for_ranking.keys())}")
    df_ranks = pd.DataFrame(data_for_ranking)
    
    # Calculate ranks for each row (lower values get better ranks, i.e., rank 1)
    ranks_df = df_ranks.rank(axis=1, method='min', ascending=True)
    
    # Calculate average ranks
    avg_ranks = ranks_df.mean()
    
    return avg_ranks.tolist(), list(avg_ranks.index)

# Step 3: Create critical difference diagram for method-sample size combinations
def create_cd_diagram_combined(metric_name, avg_ranks, method_names, num_datasets, output_dir="analysis/end_to_end_eval/Results/cd_figures/"):
    """
    Create and save a critical difference diagram for method-sample size combinations.
    
    Args:
        metric_name (str): Name of the metric (e.g., 'ECE', 'MCE')
        avg_ranks (list): List of average ranks
        method_names (list): List of method names
        num_datasets (int): Number of datasets used
        output_dir (str): Directory to save the diagram
    """
    if len(avg_ranks) == 0 or len(method_names) == 0:
        print(f"    Skipping {metric_name}: No data available")
        return
    
    print(f"    Creating combined CD diagram for {metric_name}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute Critical Difference
    cd = compute_CD(avg_ranks, num_datasets)
    print(f"    Critical Difference: {cd:.4f}")
    
    # Create the plot with larger width to accommodate more methods
    plt.figure(figsize=(14, 5))
    graph_ranks(avg_ranks, method_names, cd=cd, width=8, textspace=2)
    
    # Add title with metric name
    plt.title(f"Critical Difference Diagram - {metric_name.upper()}\nMethod-Sample Size Combinations (Friedman/Nemenyi Test)", 
              fontsize=14, pad=20)
    
    # Save the plot
    filename = f"{output_dir}cd_diagram_{metric_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"    Critical difference diagram for {metric_name} saved as {filename}")
    
    # Close the figure to free memory
    plt.close()

# Step 4: Main execution
def main():
    """
    Main function to create critical difference diagrams with method-sample size combinations.
    """
    # Path to the CSV file
    csv_path = "analysis/end_to_end_eval/Results/combined_data_for_comparison.csv"
    
    try:
        # Load data from CSV with combined method-sample size approach
        print("Loading data from CSV...")
        data_by_metric = load_data_from_csv_combined(csv_path)
        
        # Metric display names
        metric_display_names = {
            'ece': 'ECE',
            'mce': 'MCE', 
            'confece': 'ConfECE',
            'log_loss': 'Log Loss',
            'brier_score': 'Brier Score'
        }
        
        print(f"\n{'='*60}")
        print("Creating Combined Critical Difference Diagrams")
        print(f"{'='*60}")
        
        # Process each metric
        for metric, metric_data in data_by_metric.items():
            print(f"\n  Processing {metric.upper()}...")
            
            # Calculate average ranks
            avg_ranks, method_names = calculate_average_ranks_combined(metric_data)
            
            if len(avg_ranks) > 0:
                # Get number of datasets 
                num_datasets = min(len(data) for data in metric_data.values() if len(data) > 0)
                
                print(f"    Methods: {method_names}")
                print(f"    Average ranks: {[f'{rank:.3f}' for rank in avg_ranks]}")
                print(f"    Number of datasets: {num_datasets}")
                
                # Create CD diagram
                display_name = metric_display_names.get(metric, metric.upper())
                create_cd_diagram_combined(display_name, avg_ranks, method_names, num_datasets)
            else:
                print(f"    No valid data found for {metric}")
                
        print("\n" + "="*60)
        print("All combined critical difference diagrams have been generated!")
        print("Files created:")
        for metric in ['ECE', 'MCE', 'ConfECE', 'Log Loss', 'Brier Score']:
            filename = f"cd_diagram_{metric.lower().replace(' ', '_')}_combined.png"
            print(f"  - {filename}")
            
        print("\nEach diagram shows:")
        print("  - RandomSearch10, RandomSearch30, RandomSearch50")
        print("  - SmartCal10, SmartCal30, SmartCal50") 
        print("  - BetaCal, TempScaling")
        
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {csv_path}")
        print("Please ensure the file exists and the path is correct.")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
