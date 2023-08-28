from sklearn.svm import SVC
from ..BaseClassifier.base_classifier import Classifier

class SVModel(Classifier):
    def __init__(self) -> None:
        self._classifier = None

    def get_X_and_y(self, target, X_features):
        return super().get_X_and_y(target, X_features)
    
    def preprocess(self, X_train, y_train, X_test, y_test):
        return super().preprocess(X_train, y_train, X_test, y_test)