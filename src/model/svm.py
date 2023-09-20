from src.model.base_classifier import Classifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SVModel(Classifier):
    def __init__(self, C, kernel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__model = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel, C=C)
        )
    
    def train_model(self, X_train, y_train):
        y_train_transformed = self._label_encoder.fit_transform(y_train)

        self.__model.fit(X_train, y_train_transformed)
    
    def create_confusion_matrix(self, X_test, y_test):
        y_pred =  self.__model.predict(X_test)
        y_test = self._label_encoder.fit_transform(y_test)
        cm = confusion_matrix(y_test, y_pred)

        # Create a heatmap of the confusion matrix
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.close(fig)
        return fig

    def classify(self, X):
        y_pred = self.__model.predict(X)
        return y_pred

    def save_model(self, model_filepath):
        if self.__model is not None:
            with open(model_filepath, 'wb') as model:
                pickle.dump(self.__model, model)
            print("Model saved successfully.")
        else:
            print("No trained model available to save.")
