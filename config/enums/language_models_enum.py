from transformers import AutoModelForSequenceClassification
import fasttext

from enum import Enum


class ModelType(Enum):
    BERT = "BERT"
    WordEmbeddingModel = "WordEmbeddingModel"


class LanguageModelsEnum(Enum):
    # MOBILEBERT = ("google/mobilebert-uncased", AutoModelForSequenceClassification, ModelType.BERT)
    TINYBERT = ("huawei-noah/TinyBERT_General_4L_312D", AutoModelForSequenceClassification, ModelType.BERT)
    FASTTEXT = ("fasttext", None, ModelType.WordEmbeddingModel)  # No pretrained model class for FastText

    def __init__(self, model_name, model_class, model_type):
        self._model_name = model_name
        self._model_class = model_class
        self._model_type = model_type

    @property
    def model_type(self):
        """Return the broader model family/type (e.g., BERT or WordEmbeddingModel)."""
        return self._model_type

    @property
    def model_name(self):
        """Return the pretrained model identifier or 'fasttext' for FastText models."""
        return self._model_name

    def __call__(self, *args, **kwargs):
        """
        Make the enum callable to initialize the corresponding model.

        Returns:
            - A pretrained Hugging Face model for BERT-based models.
            - A placeholder FastText model for fasttext.
        """
        if self._model_class:
            return self._model_class.from_pretrained(self._model_name, *args, **kwargs)
        else:
            return fasttext.FastText._FastText()  # Placeholder FastText model

    def __str__(self):
        """Returns the string representation of the model name."""
        return self._model_name
    
    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

