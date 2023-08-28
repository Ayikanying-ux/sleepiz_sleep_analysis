import pickle
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class Classifier(ABC):
    def __init__(self, model_filepath, test_size=0.2) -> None:
        self.test_size = test_size
        self.model_filepath = model_filepath
    
    @abstractmethod
    def get_X_and_y(self, target, X_features):
        pass

    def train_test_split(self):
        y, X = self.get_X_and_y()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    @abstractmethod
    def preprocess(self, X_train, y_train, X_test, y_test):
        pass
    
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate(self, X_test):
        pass

    def save_model(self):
        if self.classifier is not None:
            with open(self.model_filepath, 'wb') as model:
                pickle.dump(self.classifier, model)
            print("Model saved successfully.")
        else:
            print("No trained model available to save.")