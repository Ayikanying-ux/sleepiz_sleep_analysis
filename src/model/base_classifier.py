from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Any
import numpy as np

class Classifier(ABC):
    def __init__(self):
        self._label_encoder =LabelEncoder()

    @abstractmethod
    def train_model(self, X_train, y_train):
        pass

    @abstractmethod
    def create_confusion_matrix(self, X_test, y_test):
        pass

    @abstractmethod
    def save_model(self, model_filepath):
        pass

    @staticmethod
    def train_test_split(X:np.array, Y:np.array, test_size = 0.2) -> Tuple[np.array, np.array, np.array, np.array]:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)
        return X_train, X_test, Y_train, Y_test

    def fit_transform(self, y_test):
        return self._label_encoder.fit_transform(y_test)