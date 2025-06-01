from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import joblib

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


# Initialize configuration manager
config_manager = ConfigurationManager()

# Create models directory if it doesn't exist
os.makedirs(config_manager.meta_models_path, exist_ok=True)

# Configuration
CONFIG = {
    'meta_data_path': config_manager.meta_data_file,
    'insights_folder_path': config_manager.insights_folder_path,
    'models_folder_path': config_manager.meta_models_path,
    'test_size': 0.1,
    'random_state': 42,
    'target_col': 'Best_Cal',
    'N_value': 5
}

# Define columns for results
RESULTS_COLS = ["Dataset", "Model", "Split", f"Recall@N:{CONFIG['N_value']}", "MRR",
                "Accuracy", "F1_micro", "F1_macro", "F1_weighted",
                "Recall_micro", "Recall_macro", "Recall_weighted",
                "Precision_micro", "Precision_macro", "Precision_weighted"]

# Define models with optimized hyperparameters
MODEL_LIST = {
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=3,
            min_samples_leaf=1,
            criterion='gini'
        ),
        n_estimators=250,
        learning_rate=0.08,
        random_state=CONFIG['random_state']
    )
}

def compute_metrics(y_true, y_pred):
    """Compute recall@N and MRR metrics"""
    recall_n_scores, mrr_scores, prediction_lengths = [], [], []
    for true_class, predicted_classes in zip(y_true, y_pred):
        prediction_lengths.append(len(predicted_classes))
        recall_n_scores.append(int(true_class in predicted_classes))
        if true_class in predicted_classes:
            rank = predicted_classes.index(true_class) + 1
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0)
    avg_preds = np.mean(prediction_lengths)
    return (
        np.mean(recall_n_scores),
        np.mean(mrr_scores),
        avg_preds
    )

def compute_standard_metrics(y_true, y_pred):
    """Compute standard classification metrics"""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "Precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0)
    }

def generate_insights(df_metric, metric, output_dir, target_col):
    """Generate visualizations and insights for a metric dataset"""
    safe_metric = re.sub(r'[^\w\-_.]', '_', str(metric))
    
    # Class distribution plot
    distplot_dir = os.path.join(output_dir, "class_distributions")
    os.makedirs(distplot_dir, exist_ok=True)
    
    class_counts = pd.Series(df_metric[target_col]).value_counts().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    
    total_count = class_counts.sum()
    class_counts_df = pd.DataFrame({
        'Metric': metric,
        'Class': class_counts.index,
        'Count': class_counts.values
    })
    class_counts_df['Probabilities'] = class_counts_df['Count'] / total_count
    class_counts_df['Sum'] = total_count
    
    # Save class distribution to combined CSV
    csv_path = os.path.join(distplot_dir, "all_class_distributions.csv")
    if os.path.exists(csv_path):
        # Append to existing file
        class_counts_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        # Create new file
        class_counts_df.to_csv(csv_path, index=False)
    
    sns.barplot(x='Class', y='Count', data=class_counts_df, hue='Class', palette="Set2", legend=False)
    
    plt.title(f"Class Distribution - {metric}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    
    for i, v in enumerate(class_counts.values):
        plt.text(i, v + 2, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(distplot_dir, f"{safe_metric}_class_distribution.png"))
    plt.close()
    
    # Feature correlation heatmap
    heatmap_dir = os.path.join(output_dir, "feature_correlation_heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)
    feature_corr = df_metric.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(feature_corr, cmap='coolwarm', center=0, annot=False)
    plt.title(f"Feature Correlation Heatmap - {metric}")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmap_dir, f"{safe_metric}_feature_correlation_heatmap.png"))
    plt.close()
    
    # Pairwise outperform ratio heatmap
    ratio_dir = os.path.join(output_dir, "pairwise_ratio_heatmaps")
    os.makedirs(ratio_dir, exist_ok=True)
    counts = df_metric[target_col].value_counts()
    calibrators = counts.index.tolist()
    matrix = pd.DataFrame(index=calibrators, columns=calibrators, dtype=float)
    
    for i in calibrators:
        for j in calibrators:
            if i == j:
                matrix.loc[i, j] = np.nan
            else:
                total = counts[i] + counts[j]
                matrix.loc[i, j] = counts[i] / total if total != 0 else 0
    
    plt.figure(figsize=(20, 12))
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                     cbar_kws={'label': 'Outperformance Ratio'}, linewidths=0.5, linecolor='white', annot_kws={"size": 20})
    plt.title(f"Calibrators Performing Comparison - {metric}", fontsize=25, pad=20)
    plt.xlabel("Main Calibrator", fontsize=20)
    plt.ylabel("Calibrator Outperforming", fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    cbar = ax.collections[0].colorbar
    cbar.set_label('Outperformance Ratio', fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(ratio_dir, f"{safe_metric}_pairwise_ratio_heatmap.png"))
    plt.close()

def save_results_incrementally(results, filepath):
    """Save results incrementally, appending to existing file if it exists"""
    df_new = pd.DataFrame(results)[RESULTS_COLS]
    
    if os.path.exists(filepath):
        # Read existing results and append new ones
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(filepath, index=False)
    return df_combined

def train_and_evaluate_models(df_metric, metric_name):
    """Train and evaluate models on a metric dataset"""
    results = []
    results_file = f"meta_models_training/insights/model_performance_analysis/model_performance_metrics.csv"
    
    # Create metric-specific models directory
    metric_models_dir = os.path.join(CONFIG['models_folder_path'], metric_name)
    os.makedirs(metric_models_dir, exist_ok=True)
    
    # Create a copy to avoid SettingWithCopyWarning
    df_metric = df_metric.copy()
    
    # Drop target column and create dummy variables for remaining categorical columns
    X = pd.get_dummies(df_metric.drop(columns=[CONFIG['target_col']]), drop_first=True)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_metric[CONFIG['target_col']])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state']
    )
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_train_labels = label_encoder.inverse_transform(y_train)
    
    for model_name, model_proto in MODEL_LIST.items():
        print(f"\n🔁 Evaluating Model: {model_name}")
        
        try:
            clf_multi = clone(model_proto)
            clf_multi.fit(X_train, y_train)
            
            # Save the trained model, label encoder, and feature names
            model_path = os.path.join(metric_models_dir, f"{model_name}.joblib")
            encoder_path = os.path.join(metric_models_dir, f"label_encoder.joblib")
            
            joblib.dump(clf_multi, model_path)
            joblib.dump(label_encoder, encoder_path)
            print(f"💾 Saved model and encoder for {model_name} on {metric_name}")
            
            probs_test = clf_multi.predict_proba(X_test)
            probs_train = clf_multi.predict_proba(X_train)
            
            print(f"\n⭐ Testing Top-{CONFIG['N_value']}")
            
            def topn_preds_fn(probs):
                return [
                    list(label_encoder.inverse_transform(np.argsort(row)[::-1][:CONFIG['N_value']]))
                    for row in probs
                ]
            
            # Train performance
            train_topn_preds = topn_preds_fn(probs_train)
            recall_n_topn_train, mrr_topn_train, avg_topn_train = compute_metrics(y_train_labels, train_topn_preds)
            y_train_pred_flat = clf_multi.predict(X_train)
            standard_metrics_topn_train = compute_standard_metrics(y_train, y_train_pred_flat)
            
            print(f"✅ Top-{CONFIG['N_value']} Train ({model_name}) | Recall@N: {recall_n_topn_train:.4f}, MRR: {mrr_topn_train:.4f}")
            
            results.append({
                "Dataset": metric_name,
                "Model": model_name,
                "Split": "Train",
                f"Recall@N:{CONFIG['N_value']}": round(recall_n_topn_train, 4),
                "MRR": round(mrr_topn_train, 4),
                **{k: round(v, 4) for k, v in standard_metrics_topn_train.items()}
            })
            
            # Test performance
            test_topn_preds = topn_preds_fn(probs_test)
            recall_n_topn_test, mrr_topn_test, avg_topn_test = compute_metrics(y_test_labels, test_topn_preds)
            y_test_pred_flat = clf_multi.predict(X_test)
            standard_metrics_topn_test = compute_standard_metrics(y_test, y_test_pred_flat)
            
            print(f"✅ Top-{CONFIG['N_value']} Test ({model_name}) | Recall@N: {recall_n_topn_test:.4f}, MRR: {mrr_topn_test:.4f}")
            
            results.append({
                "Dataset": metric_name,
                "Model": model_name,
                "Split": "Test",
                f"Recall@N:{CONFIG['N_value']}": round(recall_n_topn_test, 4),
                "MRR": round(mrr_topn_test, 4),
                **{k: round(v, 4) for k, v in standard_metrics_topn_test.items()}
            })
            
            # Save results incrementally after each model evaluation
            save_results_incrementally(results, results_file)
            print(f"💾 Saved results for {model_name} on {metric_name}")
            
        except Exception as e:
            print(f"⚠️ Training failed for {model_name}: {e}")
    
    return results

def main():
    """Main function to process meta data and train models"""
    # Create necessary directories
    optimization_dir = os.path.join(CONFIG['insights_folder_path'], "model_performance_analysis")
    os.makedirs(optimization_dir, exist_ok=True)
    
    # Clear the results file at the start
    results_file = f"meta_models_training/insights/model_performance_analysis/model_performance_metrics.csv"
    if os.path.exists(results_file):
        os.remove(results_file)
    
    # Clear the class distributions file at the start
    class_dist_file = os.path.join(CONFIG['insights_folder_path'], "class_distributions", "all_class_distributions.csv")
    if os.path.exists(class_dist_file):
        os.remove(class_dist_file)
    
    # Load the original meta data
    try:
        df = pd.read_csv(CONFIG['meta_data_path'])
    except Exception as e:
        print(f"Error loading input file {CONFIG['meta_data_path']}: {e}")
        return
    
    # Drop unused columns
    for col in ['Dataset_name', 'Model_name']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    # Ensure calibration column exists
    assert 'Calibration_metric' in df.columns, "Missing 'Calibration_metric' column in dataset"
    
    # Process each calibration metric
    all_results = []
    for metric in df['Calibration_metric'].unique():
        print(f"\n=== Processing metric: {metric} ===")
        
        # Create metric-specific dataset
        df_metric = df[df['Calibration_metric'] == metric].copy()
        df_metric.drop(columns='Calibration_metric', inplace=True)
        df_metric.reset_index(drop=True, inplace=True)
        
        # Generate insights
        generate_insights(df_metric, metric, CONFIG['insights_folder_path'], CONFIG['target_col'])
        
        # Train and evaluate models
        metric_results = train_and_evaluate_models(df_metric, metric)
        all_results.extend(metric_results)
    
    print("\nProcessing complete!")
    print(f"Results saved to: meta_models_training/insights/model_performance_analysis/")

if __name__ == "__main__":
    main() 