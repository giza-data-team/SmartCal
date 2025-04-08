import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from math import log
from .preprocessor_base import Preprocessor
from Package.src.SmartCal.utils.timer import time_operation
from Package.src.SmartCal.config.configuration_manager.configuration_manager import ConfigurationManager

config_manager = ConfigurationManager()

class CustomLabelEncoder(LabelEncoder):
    def transform(self, y):
        seen_classes = np.append(self.classes_, '<UNK>')
        unseen_label = np.where(seen_classes == '<UNK>')[0][0]
        return np.array([
            np.where(seen_classes == label)[0][0] 
            if label in self.classes_ else unseen_label 
            for label in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

class TabularPreprocessor(Preprocessor):
    def __init__(self, dataset_name, metadata_path=config_manager.config_language,logs=False):
        super().__init__(dataset_name=dataset_name, metadata_path=metadata_path, logs=logs)
        
        self.dataset_name = dataset_name
        self.logs = logs
        self.metadata_path = metadata_path
        self.timing = {} 
    
        self.config = self.load_dataset_config()
        
        self.column_types = {
            'categorical': [], 
            'numerical': [], 
            'datetime': [],
            'target': self.config['Target']
            }
        
        self.label_encoders = {}
        self.fitted = False
        self.imputers = {}
        self.datetime_features = {}
        self.target_encoder = None
    
    def validate_config_columns(self, config):
        """Validate required columns for tabular preprocessing"""
        required_columns = {'Dataset', 'Target'}
        missing_columns = required_columns - set(config.keys())
        if missing_columns:
            raise ValueError(f"Missing required columns in config: {missing_columns}")

    def select_scaler(self, data):
        # Select the best scaling technique based on data distribution (outliers/skewness).
        skewness = data.skew().abs().mean()
        has_outliers = any((data - data.mean()).abs() > 3 * data.std())
        
        if has_outliers:
            return RobustScaler()
        elif skewness > 1:
            return MinMaxScaler()
        else:
            return StandardScaler()
        
    def detect_column_types(self, data):
        """Detect and categorize column types"""
        try:
            num_samples = len(data)
            log_num_samples = log(num_samples)

            self.column_types['categorical'] = []
            self.column_types['numerical'] = []
            self.column_types['datetime'] = []
            
            for column in data.columns:
                if column == self.column_types['target']:
                    continue
                    
                # Check if column is datetime type
                if pd.api.types.is_datetime64_any_dtype(data[column]):
                    self.column_types['datetime'].append(column)
                    continue
                    
                # Categorize based on unique values
                unique_values = data[column].nunique()
                if unique_values < log_num_samples or data[column].dtype in ['object', 'category']:
                    self.column_types['categorical'].append(column)
                else:
                    self.column_types['numerical'].append(column)
                    
        except Exception as e:
            if self.logs:
                self.log_preprocessing_info(
                    "Column Type Detection Error",
                    error=str(e)
                )
            raise ValueError(f"Error in detect_column_types: {str(e)}")

    def handle_missing_values(self, data):
        """Handle missing values in the dataset"""
        data = data.replace([np.inf, -np.inf], np.nan)
        
        for col_type, columns in self.column_types.items():
            if isinstance(columns, list):
                for col in columns:
                    if col not in data.columns:
                        continue
                    
                    if col not in self.imputers:
                        if col_type in ['categorical']:
                            self.imputers[col] = SimpleImputer(strategy='most_frequent')
                        else:
                            strategy = 'median' if abs(data[col].skew()) > 1 else 'mean'
                            self.imputers[col] = SimpleImputer(strategy=strategy)
                    
                    try:
                        imputed_values = self.imputers[col].fit_transform(data[[col]])
                        data[col] = imputed_values.ravel()
                    except Exception as e:
                        if self.logs:
                            self.log_preprocessing_info(
                                "Imputation Error",
                                column=col,
                                error=str(e),
                                fallback="Using most_frequent strategy"
                            )
                        self.imputers[col] = SimpleImputer(strategy='most_frequent')
                        imputed_values = self.imputers[col].fit_transform(data[[col]])
                        data[col] = imputed_values.ravel()
        
        return data

    def process_datetime(self, data, fit=True):
        if self.column_types['datetime']:
            for col in self.column_types['datetime']:
                if fit:
                    data[col] = pd.to_datetime(data[col])
                    self.datetime_features[col] = {
                        'min_date': data[col].min(),
                        'max_date': data[col].max()
                    }
                
                data[f'{col}_year'] = data[col].dt.year
                data[f'{col}_month'] = data[col].dt.month
                data[f'{col}_day'] = data[col].dt.day
                data[f'{col}_dayofweek'] = data[col].dt.dayofweek
                data[f'{col}_days_since_min'] = (data[col] - self.datetime_features[col]['min_date']).dt.days
                
                data = data.drop(columns=[col])
        
        return data

    def encode_categorical(self, data, fit):
        """Encode categorical columns"""
        data = data.copy()
        categorical_columns = [col for col in self.column_types['categorical'] if col in data.columns]
        
        for col in categorical_columns:
            try:
                if fit:
                    le = CustomLabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        data[col] = le.transform(data[col].astype(str))
                    else:
                        le = CustomLabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        self.label_encoders[col] = le
            except Exception as e:
                if self.logs:
                    self.log_preprocessing_info(
                        "Encoding Error",
                        column=col,
                        error=str(e)
                    )
                raise ValueError(f"Error encoding column '{col}': {str(e)}")
        
        return data

    def normalize_numerical(self, data, fit):
        if self.column_types['numerical']:
            numerical_data = data[self.column_types['numerical']].reindex(columns=self.column_types['numerical'])
            
            if fit:
                self.scaler = self.select_scaler(numerical_data)
                scaled_data = self.scaler.fit_transform(numerical_data)
            else:
                scaled_data = self.scaler.transform(numerical_data)
                
            for i, col in enumerate(self.column_types['numerical']):
                data[col] = scaled_data[:, i]
                
        return data

    def remove_empty_data(self, data):
        """Remove empty columns from the dataset"""
        data = data.copy()
        
        def is_empty_column(col):
            if pd.api.types.is_numeric_dtype(data[col]):
                return data[col].isna().all()
            return data[col].isna().all() or (data[col].astype(str).str.strip() == '').all()
        
        empty_cols = [col for col in data.columns if is_empty_column(col)]
        
        if empty_cols:
            if self.logs:
                self.log_preprocessing_info(
                    "Empty Columns Removal",
                    removed_columns=empty_cols
                )
            data = data.drop(columns=empty_cols)
        
        return data

    @time_operation
    def fit_transform(self, data):
        data = data.copy()
        target_col = self.column_types['target']
        
        target = data[target_col].copy()
        data = self.remove_empty_data(data)
        self.detect_column_types(data)
        
        if self.logs:
            self.log_preprocessing_info(
                "Data Types Detection",
                categorical_columns=self.column_types['categorical'],
                numerical_columns=self.column_types['numerical'],
                datetime_columns=self.column_types['datetime'],
                target_column=target_col
            )
        
        data = self.handle_missing_values(data)
        data = self.process_datetime(data, fit=True)
        data = self.encode_categorical(data, fit=True)
        data = self.normalize_numerical(data, fit=True)
        
        if not pd.api.types.is_numeric_dtype(target) or target.dtype == 'object':
            self.target_encoder = CustomLabelEncoder()
            target = self.target_encoder.fit_transform(target.astype(str))
        else:
            # Ensure consecutive integer labels
            unique_labels = sorted(target.unique())
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.target_encoder = label_map
            target = target.map(label_map)
        
        self.fitted = True
        
        X = data.drop(columns=[target_col])
        y = target
        
        if self.logs:
            self.log_preprocessing_info(
                "Preprocessing Complete",
                final_shape=data.shape,
                num_features=data.shape[1] - 1,  # excluding target
                num_samples=len(data),
                target_unique_values=len(set(target)),
            )
        return X, y

    @time_operation
    def transform(self, data):
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before calling transform")
        
        data = data.copy()
        target_col = self.column_types['target']
        target = data[target_col].copy()
        
        if self.logs:
            self.log_preprocessing_info(
                "Transform Start",
                input_shape=data.shape,
                columns=list(data.columns),
                target_column=target_col
            )
        
        data = self.remove_empty_data(data)
        
        data = self.handle_missing_values(data)
        data = self.process_datetime(data, fit=False)
        data = self.encode_categorical(data, fit=False)
        data = self.normalize_numerical(data, fit=False)

        if self.target_encoder is not None:
            if isinstance(self.target_encoder, dict):
                # For numeric labels that were mapped
                target = target.map(self.target_encoder)
            else:
                # For string labels that were encoded
                target = self.target_encoder.transform(target.astype(str))

        X = data.drop(columns=[target_col])
        y = target
        
        if self.logs:
            self.log_preprocessing_info(
                "Transform Complete",
                final_shape=data.shape,
                num_features=data.shape[1] - 1,  # excluding target
                num_samples=len(data),
                target_unique_values=len(set(target)),
            )
        return X, y
        
    def get_timing(self):
        return self.timing