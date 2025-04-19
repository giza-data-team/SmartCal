import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler

from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager


# Load configuration
config = ConfigurationManager()

CONFIG = {
    'test_size': 0.1,
    'target_col': 'Best_Cal',
    'scaler': 1,
    'ordinal': 1
}

# 1. Load data
df = pd.read_csv(config.meta_data_file)

# 2. Separate features & target
X = df.drop(columns=[CONFIG['target_col']])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[CONFIG['target_col']])
joblib.dump(label_encoder, config.meta_label_encoder_path)
print("Label encoder saved")

# Optional: Apply scaler and ordinal preprocessing
if CONFIG['scaler'] == 1:
    scaler_model = StandardScaler()
    X_scaled = scaler_model.fit_transform(X.select_dtypes(include=np.number))
    joblib.dump(scaler_model, config.meta_scaler_path)
    print("Scaler model saved")
else:
    X_scaled = X.select_dtypes(include=np.number)

if CONFIG['ordinal'] == 1:
    ordinal_model = OrdinalEncoder()
    X_categorical = X.select_dtypes(exclude=np.number)
    X_encoded = ordinal_model.fit_transform(X_categorical)
    joblib.dump(ordinal_model, config.meta_ordinal_encoder_path)
    print("Ordinal model saved")
else:
    X_encoded = X.select_dtypes(exclude=np.number)

# Combine numeric and encoded categorical data
X_processed = np.hstack([X_scaled, X_encoded])

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=CONFIG['test_size'],
    stratify=y,
    random_state=config.random_seed
)

# 4. Undersample & Train One-vs-All models
undersampled_data_splits = {}
for class_label in np.unique(y_train):
    y_train_binary = np.where(y_train == class_label, 1, 0)
    rus = RandomUnderSampler(random_state=config.random_seed)
    X_train_resampled, y_train_resampled = rus.fit_resample(X_train, y_train_binary)
    undersampled_data_splits[class_label] = (X_train_resampled, y_train_resampled)

models = {}
for class_label, (X_train_resampled, y_train_resampled) in undersampled_data_splits.items():
    model = LogisticRegression(
        random_state=config.random_seed,
        max_iter=5000, C=0.1, penalty='l2',
        solver='newton-cg'
    )
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_train_resampled)
    f1 = f1_score(y_train_resampled, y_pred)
    models[class_label] = model
    print(f"Class {class_label}: F1-score = {f1:.4f}")

# 5. Save One-vs-All model
model_filename = config.meta_model_path
joblib.dump(models, model_filename)
print(f"Meta-model saved")
