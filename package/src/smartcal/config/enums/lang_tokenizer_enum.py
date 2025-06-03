from enum import Enum


class LanguageModelTokenizer(Enum):
    BERT_TOKENIZER = "bert-base-uncased"

    def __str__(self):
        return self.value
