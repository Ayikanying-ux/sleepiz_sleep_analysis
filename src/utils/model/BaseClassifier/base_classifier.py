import pickle
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split

class Classifier(ABC):
    def __init__(self, dataframe, test_size=0.2) -> None:
        self.test_size = test_size
        self.dataframe = dataframe
    
    @abstractmethod
    def get_X_and_y(self, features, target):
        self.dataframe['features'] = self.dataframe['features'].apply(lambda x: np.array(x, dtype=float))

        X = self.dataframe[features].to_list()
        y = self.dataframe[target]
        return X, y

    def train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    @abstractmethod
    def train_model(self, X_train, y_train):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    def save_model(self, classifier, model_filepath):
        if classifier is not None:
            with open(model_filepath, 'wb') as model:
                pickle.dump(classifier, model)
            print("Model saved successfully.")
        else:
            print("No trained model available to save.")