#!/usr/bin/env python3
"""
Complete script to create a comprehensive comparison CSV of calibration methods with ALL datasets.
Format: dataset_name, model, n, randomsearch_ece, smartcal_ece, beta_ece, temp_scaling_ece, etc.
for all metrics: ECE, MCE, ConfECE, Log Loss, Brier Score
The 'n' column contains values 10, 30, or 50 for randomsearch parameters.
Automatically fixes ConfECE columns by extracting first value from lists.
"""

import pandas as pd
import numpy as np
import ast
import os
from pathlib import Path

def extract_first_value_from_list_string(value):
    """Extract the first value from a list string like '[1.5e-05, 2.0e-05, ...]'"""
    if pd.isna(value) or value == '':
        return np.nan
    
    try:
        # Parse the string as a Python list
        parsed_list = ast.literal_eval(value)
        if isinstance(parsed_list, list) and len(parsed_list) > 0:
            return parsed_list[0]
        else:
            return np.nan
    except:
        # If parsing fails, try to return the value as is
        return value

def load_and_process_data():
    """Load and process all calibration data."""
    results_dir = Path("experiments/end_to_end_eval/Results")
    
    # Load SmartCal results
    print("Loading SmartCal results...")
    smartcal_df = pd.read_csv(results_dir / "smartcal_results.csv")
    smartcal_df = smartcal_df.rename(columns={'dataset_name': 'dataset', 'model_name': 'model'})
    
    # Process SmartCal data the same way as Random Search - keep individual rows for each n
    # Select columns and rename Total_Iterations to n to match Random Search structure
    smartcal_processed = smartcal_df[['dataset', 'model', 'Total_Iterations', 'test_ece', 'test_mce', 'test_conf_ece', 'test_log_loss', 'test_brier_score']].copy()
    
    # Rename columns for consistency
    smartcal_processed = smartcal_processed.rename(columns={
        'Total_Iterations': 'n',
        'test_ece': 'smartcal_ece',
        'test_mce': 'smartcal_mce', 
        'test_conf_ece': 'smartcal_confece',
        'test_log_loss': 'smartcal_log_loss',
        'test_brier_score': 'smartcal_brier_score'
    })
    
    # Apply ConfECE fixing to SmartCal (same as Random Search)
    smartcal_processed['smartcal_confece'] = smartcal_processed['smartcal_confece'].apply(extract_first_value_from_list_string)
    
    print("Loading Random Search results...")
    random_df = pd.read_csv(results_dir / "baseline_random_search_results.csv")
    random_df = random_df.rename(columns={'Dataset Name': 'dataset', 'Classification Model': 'model'})
    
    # Process Random Search data by n value (10, 30, 50) - keep n column
    random_processed = random_df[['dataset', 'model', 'n', 'test_ece', 'test_mce', 'test_conf_ece', 'test_loss', 'test_brier_score']].copy()
    
    # Rename columns for Random Search
    random_processed = random_processed.rename(columns={
        'test_ece': 'randomsearch_ece', 
        'test_mce': 'randomsearch_mce', 
        'test_conf_ece': 'randomsearch_confece', 
        'test_loss': 'randomsearch_log_loss', 
        'test_brier_score': 'randomsearch_brier_score'
    })
    
    # Apply ConfECE fixing to Random Search
    random_processed['randomsearch_confece'] = random_processed['randomsearch_confece'].apply(extract_first_value_from_list_string)
    
    print("Loading Dump Calibrator results...")
    dump_df = pd.read_csv(results_dir / "dump_calibrator_results.csv", on_bad_lines='skip')
    
    # Process dump calibrator results - get ALL metrics from main file
    beta_df = dump_df[dump_df['calibration_algorithm'] == 'BETA'][['dataset_name', 'classification_model', 'calibrated_test_ece', 'calibrated_test_mce', 'calibrated_test_conf_ece', 'calibrated_test_loss', 'calibrated_test_brier_score']].copy()
    temp_df = dump_df[dump_df['calibration_algorithm'] == 'TEMPERATURESCALING'][['dataset_name', 'classification_model', 'calibrated_test_ece', 'calibrated_test_mce', 'calibrated_test_conf_ece', 'calibrated_test_loss', 'calibrated_test_brier_score']].copy()
    
    # Group by dataset and model, then take mean across all trials
    # For ConfECE, extract first value from lists before taking mean
    beta_df_grouped = beta_df.groupby(['dataset_name', 'classification_model']).agg({
        'calibrated_test_ece': 'mean',
        'calibrated_test_mce': 'mean', 
        'calibrated_test_conf_ece': lambda x: x.apply(extract_first_value_from_list_string).mean(),
        'calibrated_test_loss': 'mean',
        'calibrated_test_brier_score': 'mean'
    }).reset_index()
    
    temp_df_grouped = temp_df.groupby(['dataset_name', 'classification_model']).agg({
        'calibrated_test_ece': 'mean',
        'calibrated_test_mce': 'mean',
        'calibrated_test_conf_ece': lambda x: x.apply(extract_first_value_from_list_string).mean(), 
        'calibrated_test_loss': 'mean',
        'calibrated_test_brier_score': 'mean'
    }).reset_index()
    
    # Rename columns for consistency
    beta_df_grouped = beta_df_grouped.rename(columns={
        'dataset_name': 'dataset', 
        'classification_model': 'model', 
        'calibrated_test_ece': 'beta_ece', 
        'calibrated_test_mce': 'beta_mce',
        'calibrated_test_conf_ece': 'beta_confece',
        'calibrated_test_loss': 'beta_log_loss', 
        'calibrated_test_brier_score': 'beta_brier_score'
    })
    
    temp_df_grouped = temp_df_grouped.rename(columns={
        'dataset_name': 'dataset',
        'classification_model': 'model', 
        'calibrated_test_ece': 'temp_scaling_ece', 
        'calibrated_test_mce': 'temp_scaling_mce',
        'calibrated_test_conf_ece': 'temp_scaling_confece',
        'calibrated_test_loss': 'temp_scaling_log_loss', 
        'calibrated_test_brier_score': 'temp_scaling_brier_score'
    })
    
    print(f"Beta calibration: {len(beta_df_grouped)} dataset-model combinations")
    print(f"Temperature scaling: {len(temp_df_grouped)} dataset-model combinations")
    
    return smartcal_processed, random_processed, beta_df_grouped, temp_df_grouped

def determine_winners(df):
    """Determine winner (best performing method) for each metric within each n value."""
    print("Determining winners for each metric...")
    
    # Define the methods to compare
    methods = ['randomsearch', 'smartcal', 'beta', 'temp_scaling']
    
    # Define metrics where lower is better
    metrics = ['ece', 'mce', 'confece', 'log_loss', 'brier_score']
    
    for metric in metrics:
        # Get columns for this metric
        metric_cols = [f'{method}_{metric}' for method in methods if f'{method}_{metric}' in df.columns]
        
        if metric_cols:
            # For each row, find the method with minimum value
            def find_winner(row):
                values = {}
                for col in metric_cols:
                    if pd.notna(row[col]):
                        method = col.replace(f'_{metric}', '')
                        values[method] = row[col]
                
                if values:
                    min_value = min(values.values())
                    # Find all methods with the minimum value
                    winners = [method for method, value in values.items() if value == min_value]
                    
                    if len(winners) > 1:
                        return "tie"
                    else:
                        return winners[0]
                else:
                    return np.nan
            
            df[f'{metric}_winner'] = df.apply(find_winner, axis=1)
            
            # Print winner statistics
            winner_counts = df[f'{metric}_winner'].value_counts()
            print(f"{metric.upper()} winners: {dict(winner_counts)}")
        else:
            print(f"No columns found for {metric}")
    
    return df

def create_complete_comparison():
    """Create a complete comparison table with actual dataset-model combinations that exist in the data."""
    smartcal_df, random_df, beta_df, temp_df = load_and_process_data()
    
    # Get all actual dataset-model combinations from all sources
    smartcal_combinations = set(zip(smartcal_df['dataset'], smartcal_df['model']))
    random_combinations = set(zip(random_df['dataset'], random_df['model']))
    beta_combinations = set(zip(beta_df['dataset'], beta_df['model']))
    temp_combinations = set(zip(temp_df['dataset'], temp_df['model']))
    
    # Union of all actual combinations
    all_combinations = smartcal_combinations.union(random_combinations).union(beta_combinations).union(temp_combinations)
    all_combinations = sorted(list(all_combinations))
    
    print(f"Found {len(all_combinations)} actual dataset-model combinations")
    
    # Get unique datasets and models for summary
    all_datasets = sorted(set([combo[0] for combo in all_combinations]))
    all_models = sorted(set([combo[1] for combo in all_combinations]))
    
    print(f"Datasets: {len(all_datasets)} - {all_datasets}")
    print(f"Models: {len(all_models)} - {all_models}")
    
    results = []
    
    for dataset, model in all_combinations:
        # For each n value (10, 30, 50), create a separate row
        for n_value in [10, 30, 50]:
            row = {'dataset_name': dataset, 'model': model, 'n': n_value}
            
            # Random Search results for this n value
            rs_data = random_df[(random_df['dataset'] == dataset) & 
                               (random_df['model'] == model) & 
                               (random_df['n'] == n_value)]
            if not rs_data.empty:
                row['randomsearch_ece'] = rs_data['randomsearch_ece'].iloc[0]
                row['randomsearch_mce'] = rs_data['randomsearch_mce'].iloc[0]
                row['randomsearch_confece'] = rs_data['randomsearch_confece'].iloc[0]
                row['randomsearch_log_loss'] = rs_data['randomsearch_log_loss'].iloc[0]
                row['randomsearch_brier_score'] = rs_data['randomsearch_brier_score'].iloc[0]
            else:
                row['randomsearch_ece'] = np.nan
                row['randomsearch_mce'] = np.nan
                row['randomsearch_confece'] = np.nan
                row['randomsearch_log_loss'] = np.nan
                row['randomsearch_brier_score'] = np.nan
            
            # SmartCal results for this n value
            sc_data = smartcal_df[(smartcal_df['dataset'] == dataset) & 
                                 (smartcal_df['model'] == model) & 
                                 (smartcal_df['n'] == n_value)]
            if not sc_data.empty:
                row['smartcal_ece'] = sc_data['smartcal_ece'].iloc[0]
                row['smartcal_mce'] = sc_data['smartcal_mce'].iloc[0]
                row['smartcal_confece'] = sc_data['smartcal_confece'].iloc[0]
                row['smartcal_log_loss'] = sc_data['smartcal_log_loss'].iloc[0]
                row['smartcal_brier_score'] = sc_data['smartcal_brier_score'].iloc[0]
            else:
                row['smartcal_ece'] = np.nan
                row['smartcal_mce'] = np.nan
                row['smartcal_confece'] = np.nan
                row['smartcal_log_loss'] = np.nan
                row['smartcal_brier_score'] = np.nan
            
            # Beta calibration results (same for all n values)
            beta_data = beta_df[(beta_df['dataset'] == dataset) & 
                               (beta_df['model'] == model)]
            if not beta_data.empty:
                row['beta_ece'] = beta_data['beta_ece'].iloc[0]
                row['beta_mce'] = beta_data['beta_mce'].iloc[0]
                row['beta_confece'] = beta_data['beta_confece'].iloc[0]
                row['beta_log_loss'] = beta_data['beta_log_loss'].iloc[0]
                row['beta_brier_score'] = beta_data['beta_brier_score'].iloc[0]
            else:
                row['beta_ece'] = np.nan
                row['beta_mce'] = np.nan
                row['beta_confece'] = np.nan
                row['beta_log_loss'] = np.nan
                row['beta_brier_score'] = np.nan
            
            # Temperature Scaling results (same for all n values)
            temp_data = temp_df[(temp_df['dataset'] == dataset) & 
                               (temp_df['model'] == model)]
            if not temp_data.empty:
                row['temp_scaling_ece'] = temp_data['temp_scaling_ece'].iloc[0]
                row['temp_scaling_mce'] = temp_data['temp_scaling_mce'].iloc[0]
                row['temp_scaling_confece'] = temp_data['temp_scaling_confece'].iloc[0]
                row['temp_scaling_log_loss'] = temp_data['temp_scaling_log_loss'].iloc[0]
                row['temp_scaling_brier_score'] = temp_data['temp_scaling_brier_score'].iloc[0]
            else:
                row['temp_scaling_ece'] = np.nan
                row['temp_scaling_mce'] = np.nan
                row['temp_scaling_confece'] = np.nan
                row['temp_scaling_log_loss'] = np.nan
                row['temp_scaling_brier_score'] = np.nan
            
            results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Determine winners for each metric
    df = determine_winners(df)
    
    # Organize columns by metric type as requested
    base_cols = ['dataset_name', 'model', 'n']
    
    # ECE columns
    ece_cols = ['randomsearch_ece', 'smartcal_ece', 'beta_ece', 'temp_scaling_ece']
    
    # MCE columns
    mce_cols = ['randomsearch_mce', 'smartcal_mce', 'beta_mce', 'temp_scaling_mce']
    
    # ConfECE columns
    confece_cols = ['randomsearch_confece', 'smartcal_confece', 'beta_confece', 'temp_scaling_confece']
    
    # Log Loss columns
    log_loss_cols = ['randomsearch_log_loss', 'smartcal_log_loss', 'beta_log_loss', 'temp_scaling_log_loss']
    
    # Brier Score columns
    brier_cols = ['randomsearch_brier_score', 'smartcal_brier_score', 'beta_brier_score', 'temp_scaling_brier_score']
    
    # Winner columns
    winner_cols = ['ece_winner', 'mce_winner', 'confece_winner', 'log_loss_winner', 'brier_score_winner']
    
    # Final column order
    final_cols = base_cols + ece_cols + mce_cols + confece_cols + log_loss_cols + brier_cols + winner_cols
    
    # Only include columns that exist in the dataframe
    final_cols = [col for col in final_cols if col in df.columns]
    df = df[final_cols]
    
    return df

def main():
    """Main function to create the complete calibration comparison CSV."""
    try:
        print("Creating COMPLETE calibration comparison with ALL datasets...")
        print("(Automatically fixing ConfECE values by extracting first value from lists)")
        df = create_complete_comparison()
        
        # Create results directory if it doesn't exist
        output_file = "Analysis/end_to_end_eval/Results/combined_data_for_comparison.csv"
        results_dir = os.path.dirname(output_file)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            print(f"Created directory: {results_dir}")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"\nComplete comparison saved to: {output_file}")
        print(f"Shape: {df.shape}")
        print(f"Total datasets: {len(df['dataset_name'].unique())}")
        print(f"Total models: {len(df['model'].unique())}")
        print(f"Datasets: {sorted(df['dataset_name'].unique())}")
        print(f"Models: {sorted(df['model'].unique())}")
        
        # Show data availability summary for key methods
        key_methods = ['smartcal_ece', 'randomsearch_ece', 'beta_ece', 'temp_scaling_ece']
        print(f"\nData availability summary (key methods):")
        for col in key_methods:
            if col in df.columns:
                non_null = df[col].notna().sum()
                total = len(df)
                print(f"{col}: {non_null}/{total} ({non_null/total*100:.1f}%)")
        
        # Show dataset coverage by method
        print(f"\nDataset coverage by method:")
        for method in ['smartcal', 'randomsearch', 'beta', 'temp_scaling']:
            ece_col = f'{method}_ece'
            if ece_col in df.columns:
                datasets_with_data = df[df[ece_col].notna()]['dataset_name'].nunique()
                total_datasets = df['dataset_name'].nunique()
                print(f"{method}: {datasets_with_data}/{total_datasets} datasets")
        
        # Verify ConfECE columns are properly fixed
        #print(f"\n✅ ConfECE columns fixed:")
        confece_cols = [col for col in df.columns if 'confece' in col.lower()]
        for col in confece_cols[:3]:  # Show first 3 ConfECE columns
            if col in df.columns:
                non_nan_values = df[col].dropna()
                if len(non_nan_values) > 0:
                    sample_val = non_nan_values.iloc[0]
                    #print(f"{col}: {sample_val} (type: {type(sample_val).__name__})")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 