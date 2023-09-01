from utils.model.BaseClassifier.base_classifier import Classifier
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class SVModel(Classifier):
    def __init__(self, C, kernel, dataframe, test_size=0.2) -> None:
        super().__init__(dataframe, test_size)
        self.C = C
        self.kernel = kernel

    def get_X_and_y(self, features, target):
        return super().get_X_and_y(features, target)
    
    def train_test_split(self, X, y):
        return super().train_test_split(X, y)
    
    def train_model(self, X_train, y_train):
        encoder = LabelEncoder()
        y_train_transformed = encoder.fit_transform(y_train)
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel=self.kernel, C=self.C)
        )

        model.fit(X_train, y_train_transformed)
        return model
    
    def evaluate_model(self, X_test, y_test, model):
        encoder = LabelEncoder()
        y_pred =  model.predict(X_test)
        y_test = encoder.fit_transform(y_test)
        cm = confusion_matrix(y_test, y_pred)
        # Create a heatmap of the confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    def save_model(self, classifier, model_filepath):
        return super().save_model(classifier, model_filepath)