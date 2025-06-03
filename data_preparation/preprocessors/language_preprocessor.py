import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
import re

from data_preparation.preprocessors.preprocessor_base import Preprocessor
from smartcal.utils.timer import time_operation
from smartcal.config.configuration_manager.configuration_manager import ConfigurationManager
from smartcal.config.enums.language_models_enum import ModelType


config_manager = ConfigurationManager()

class LanguagePreprocessor(Preprocessor):
    def __init__(self, dataset_name, model_name, metadata_path = config_manager.config_language,
                 tokenizer_max_length = config_manager.tokenizer_max_length,
                 tokenizer = config_manager.bert_tokenizer, logs = False):
        super().__init__(dataset_name=dataset_name, metadata_path=metadata_path, logs=logs)
        self.timing = {} 
        self.model_name = model_name
        self.tokenizer_name = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.label_encoder = None
        
        self.config = self.load_dataset_config()
        
        if self.model_name == ModelType.BERT:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self.label_encoder = LabelEncoder()
        elif self.model_name == ModelType.WordEmbeddingModel:
            self.label_encoder = LabelEncoder()
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")

    def validate_config_columns(self, config):
        """Validate required columns for language preprocessing"""
        required_columns = {'Dataset', 'Target', 'Text'}
        missing_columns = required_columns - set(config.keys())
        if missing_columns:
            raise ValueError(f"Missing required columns in config: {missing_columns}")

    def clean_text(self, text):
        text = str(text)
        text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r'[^\w\s]', ' ', text) # Remove special characters
        text = re.sub(r'\d+', '', text) # Remove numbers
        text = re.sub(r'\s+', ' ', text) # Remove extra whitespace
        return text.lower().strip()

    @time_operation
    def fit_transform(self, df):
        target_col, text_col = self.config['Target'], self.config['Text']
        
        texts = df[text_col].tolist()
        target = df[target_col]
        unique_classes = sorted(target.unique())
        
        if self.model_name == ModelType.BERT:
            labels = self.label_encoder.fit_transform(target)
            inputs = self.tokenizer(
                texts,
                max_length=self.tokenizer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt")
            
            if self.logs:
                # Get the mapping of encoded labels to original classes
                label_mapping = dict(zip(range(len(self.label_encoder.classes_)), 
                                    self.label_encoder.classes_))
                
                self.log_preprocessing_info(
                    "BERT Preprocessing",
                    model_type=str(self.model_name),
                    dataset_size=len(labels),
                    max_sequence_length=self.tokenizer_max_length,
                    num_classes=len(set(labels)),
                    unique_classes=unique_classes,
                    label_encoding_mapping=label_mapping,
                    sample_text=texts[0][:100]
                )
                
            return inputs, labels
        
        elif self.model_name == ModelType.WordEmbeddingModel:
            cleaned_texts = pd.Series(texts).apply(self.clean_text)
            labels = self.label_encoder.fit_transform(target)
            processed_texts = [f"__label__{label} {text}" for label, text in zip(labels, cleaned_texts)]
            processed_labels = [f"__label__{label}" for label in labels]
            
            if self.logs:
                # Get unique processed labels
                unique_processed_labels = sorted(set(processed_labels))
                
                self.log_preprocessing_info(
                    "FastText Preprocessing",
                    dataset_size=len(processed_labels),
                    num_classes=len(set(target)),
                    original_unique_classes=unique_classes,
                    processed_unique_labels=unique_processed_labels,
                    sample_processed_text=processed_texts[0][:100]
                )
            
            return processed_texts, processed_labels

    @time_operation 
    def transform(self, df):
        target_col, text_col = self.config['Target'], self.config['Text']

        texts = df[text_col].tolist()
        target = df[target_col]
        unique_classes = sorted(target.unique())
        
        if self.model_name == ModelType.BERT:
            labels = self.label_encoder.transform(target)
            inputs = self.tokenizer(
                texts,
                max_length=self.tokenizer_max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
                )
            
            if self.logs:
                # Get the mapping of encoded labels to original classes
                label_mapping = dict(zip(range(len(self.label_encoder.classes_)), 
                                    self.label_encoder.classes_))
                
                self.log_preprocessing_info(
                    "BERT Transform",
                    model_type=str(self.model_name),
                    dataset_size=len(labels),
                    max_sequence_length=self.tokenizer_max_length,
                    num_classes=len(set(labels)),
                    unique_classes=unique_classes,
                    label_encoding_mapping=label_mapping,
                    sample_text=texts[0][:100] if texts else "No texts available"
                )
            
            return inputs, labels
        
        elif self.model_name == ModelType.WordEmbeddingModel:
            labels_fast = self.label_encoder.fit_transform(target)
            cleaned_texts = pd.Series(texts).apply(self.clean_text)
            processed_labels = [f"__label__{label}" for label in labels_fast]
            
            if self.logs:
                # Get unique processed labels
                unique_processed_labels = sorted(set(processed_labels))
                
                self.log_preprocessing_info(
                    "FastText Transform",
                    dataset_size=len(processed_labels),
                    num_classes=len(set(target)),
                    original_unique_classes=unique_classes,
                    processed_unique_labels=unique_processed_labels,
                    sample_cleaned_text=cleaned_texts.iloc[0][:100] if not cleaned_texts.empty else "No texts available"
                )
            
            return cleaned_texts, processed_labels
        
    def get_timing(self):
        return self.timing
    