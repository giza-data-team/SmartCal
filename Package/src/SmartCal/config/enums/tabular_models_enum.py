from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import RidgeClassifier

from enum import Enum

class ProbabilisticSVC(SVC):
    """Wrapper for SVC that ensures probability estimation is enabled"""
    def __init__(self):
        super().__init__(probability=True)

class TabularModelsEnum(Enum):
    """
    Enum class for representing different machine learning model architectures for tabular data.
    Provides an interface to easily select and initialize popular classifiers.
    """
    RANDOM_FOREST = RandomForestClassifier  # Random Forest Classifier
    XGBOOST = XGBClassifier  # XGBoost Classifier
    SVM = ProbabilisticSVC  # Support Vector Machine Classifier
    NAIVE_BAYES = GaussianNB  # Gaussian Naive Bayes Classifier
    DECISION_TREE = DecisionTreeClassifier  # Decision Tree Classifier
    GRADIENT_BOOSTING = GradientBoostingClassifier  # Gradient Boosting Classifier
    ADABOOST = AdaBoostClassifier  # AdaBoost Classifier

    def __call__(self, *args, **kwargs):
        """
        Make the enum callable to directly call the model function.

        Args:
            *args: Variable positional arguments to pass to the model initialization.
            **kwargs: Variable keyword arguments to pass to the model initialization.

        Returns:
            An instance of the selected model initialized with the provided arguments.
        """
        return self.value(*args, **kwargs)  # Initialize and return the model instance
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}.{self.name}"

