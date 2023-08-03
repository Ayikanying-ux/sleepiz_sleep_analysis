import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

class ModelTrainer:
    def __init__(self, data_loader, feature_extractor):
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor

    def merge_dataframe(self):
        df_snoring = pd.read_csv("dataset/csv/snoring_data.csv")
        df_non_snoring = pd.read_csv("dataset/csv/non_snoring_data.csv")
        df = pd.concat([df_snoring, df_non_snoring])

        df['Features'] = df['Features'].apply(lambda x: np.array(x.strip('[]').split()).astype(float))

        # Prepare df for model building
        df.drop(columns=['audio', 'Labels', 'Audio_Data', 'Sample_Rate'], inplace=True)
        return df

    def train_model(self):
        df = self.merge_dataframe()
        print(df)
        X = df['Features'].to_list()
        y = df['class_id']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert X_train and X_test to numpy arrays
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # Reshape X_train and X_test to 2D arrays (required for StandardScaler)
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # Initialize standard scaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.fit_transform(X_test)

        classifier = SVC(kernel='rbf')
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        report = classification_report(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        # Create a heatmap of the confusion matrix
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()
        #print(y_pred)

        print("Classification Report:")
        print(report)
        
