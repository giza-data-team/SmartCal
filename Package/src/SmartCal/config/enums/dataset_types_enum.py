from enum import Enum


class DatasetTypesEnum(Enum):
    """
    Enum class for representing different types of datasets.
    This helps in categorizing datasets by their nature, such as image, tabular, or language data.
    """
    IMAGE = "image"  # Dataset for image-related tasks
    TABULAR = "tabular"  # Dataset for tabular data tasks
    LANGUAGE = "language"  # Dataset for language-related tasks

    def __str__(self):
        """
        Returns the string representation of the Enum value.

        Returns:
            str: The value of the Enum as a string.
        """
        return self.value


    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"
