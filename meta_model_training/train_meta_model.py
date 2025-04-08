import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from Package.src.SmartCal.config.configuration_manager import ConfigurationManager
config = ConfigurationManager()

CONFIG = {
    'test_size': 0.1,
    'target_col': 'Best_Cal'
}

output_folder = 'meta_model/Results'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# 1. Load data
df = pd.read_csv(config.meta_data_file)  # double check path

# 2. Separate features & target, dummy-encode
X = df.drop(columns=[CONFIG['target_col']])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[CONFIG['target_col']])
X = pd.get_dummies(X, drop_first=True)

# Save the label encoder for future use
joblib.dump(label_encoder, os.path.join(output_folder, 'label_encoder.joblib'))

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=CONFIG['test_size'],
    stratify=y,
    random_state=config.random_seed
)

undersampled_data_splits = {}
for class_label in np.unique(y_train):
    # Create a binary classification problem for each class
    y_train_binary = np.where(y_train == class_label, 1, 0)

    # Undersample the data
    rus = RandomUnderSampler(random_state=config.random_seed)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train_binary)

    # Store the undersampled data split
    undersampled_data_splits[class_label] = (X_train_resampled, y_train_resampled)

# Train a decision tree model on each split and evaluate
models = {}
for class_label, (X_train_resampled, y_train_resampled) in undersampled_data_splits.items():
    lr_model = LogisticRegression(random_state=config.random_seed, max_iter=5000, C=0.1, penalty='l2',
                                  solver='newton-cg')
    lr_model.fit(X_train_resampled, y_train_resampled)

    # Make predictions on the training data for the current split
    y_pred = lr_model.predict(X_train_resampled)

    # Calculate the F1-score and accuracy
    f1 = f1_score(y_train_resampled, y_pred)
    models[class_label] = lr_model



    # Print the results
    print(f"Class {class_label}:")
    print(f"  F1-score: {f1:.4f}")

# Save the model using joblib
model_filename = os.path.join(output_folder, f'lr_ova_model.joblib')
joblib.dump(models, model_filename)
print(f"One vs All model saved to {model_filename}")