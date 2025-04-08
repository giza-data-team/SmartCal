# Data Preprocessing (`data_preparation/preprocessors`) 

## Global Configuration
```python
RANDOM_SEED = 42
SPLIT_RATIOS = (0.6, 0.2, 0.2)        # Train, Cal, Test
SPLIT_RATIOS_IMAGES = (0.75, 0.25)     # Train, Test
```

## Supported Modalities

### 1. Tabular Data (`data_preparation/preprocessors/tabular_preprocessor`) 
#### Configuration
- **Config File**: `data_preparation/Datasets/Tabular Datasets/Data_info.xlsx`
- **Required Format**:
```
| Dataset | Target |
|---------|---------|
| name    | column  |
```

#### Processing Pipeline
1. **Column Type Detection**
   - Categorical: unique values < log(n_samples)
   - Numerical: unique values ≥ log(n_samples)
   - Datetime: datetime64 dtype
   - Target: from config

2. **Missing Value Handling**
   - Categorical → Most frequent
   - Numerical → Mean/Median (based on skewness)
   - Empty columns → Dropped
   - Infinities → NaN → Imputed

3. **Feature Engineering**
   - Datetime → year, month, day, dayofweek, days_since_min
   - Categorical → LabelEncoded (with UNK token)
   - Target → LabelEncoded/Consecutive integers

4. **Scaling** (Auto-selected)
   - RobustScaler: If outliers present
   - MinMaxScaler: If skewness > 1
   - StandardScaler: Default case

### 2. Image Data (`data_preparation/preprocessors/images_preprocessor`) 
#### Configuration
- **Config File**: `data_preparation/Datasets/Image Datasets/Data_info.xlsx`
- **Required Format**:
```
| Dataset | Torchvision_Name | Mean | STD |
|---------|------------------|------|-----|
| name    | dataset_name     | [r,g,b] | [r,g,b] |
```

#### Hyperparameters
```python
{
    'img_size': 224,
    'batch_size': 32,
    'num_workers': 4,
    'grayscale_channels': 3,
    'crop_padding': 4
}
```

#### Processing Pipeline
1. **Training Transforms**
   - Resize → (224, 224)
   - RGB conversion
   - Random crop (padding=4)
   - Random horizontal flip
   - Normalize (per-dataset stats)

2. **Validation Transforms**
   - Resize → (224, 224)
   - RGB conversion
   - Normalize (per-dataset stats)

### 3. Language Data (`data_preparation/preprocessors/language_preprocessor`) 
#### Configuration
- **Config File**: `data_preparation/Datasets/Language Datasets/Data_info.xlsx`
- **Required Format**:
```
| Dataset | Target | Text |
|---------|--------|------|
| name    | label  | text |
```

#### Hyperparameters
```python
{
    'tokenizer_max_length': 128,
    'tokenizer': "bert-base-uncased",
    'padding': 'max_length',
    'truncation': True
}
```

#### Processing Pipeline
1. **BERT Processing**
   - Tokenization
   - Padding/Truncation
   - Label encoding
   - PyTorch tensor conversion

2. **Word Embedding Processing**
   - Text cleaning (HTML, URLs, special chars)
   - Case normalization
   - Label prefixing ("__label__")

## Usage Examples
```python
# Tabular
preprocessor = TabularPreprocessor(
    dataset_name="dataset_name",
    metadata_path="Data_info.xlsx",
    logs=False
)
X_train, y_train = preprocessor.fit_transform(train_data)
X_test, y_test = preprocessor.transform(test_data)

# Image
preprocessor = ImagePreprocessor(
    dataset_name="torchvision_dataset_name",
    metadata_path="Data_info.xlsx",
    img_size=224,
    batch_size=32,
    logs=False
)
train_loader = preprocessor.fit_transform(train_data)
test_loader = preprocessor.transform(test_data)

# Language
preprocessor = LanguagePreprocessor(
    dataset_name="dataset_name",
    model_name=ModelType.BERT,  # or ModelType.WordEmbeddingModel
    metadata_path="Data_info.xlsx",
    tokenizer_max_length=512,
    tokenizer="bert-base-uncased",
    logs=False
)
X_train, y_train = preprocessor.fit_transform(train_data)
X_test, y_test = preprocessor.transform(test_data)
```