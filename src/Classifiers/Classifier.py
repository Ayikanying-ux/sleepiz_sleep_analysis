
from abc import ABC
from sklearn.model_selection import train_test_split

class Classifier(ABC):

    def __init__(self, test_size = 0.2):
        self._test_size = test_size

    @abstractmethod
    def train(self, training_data, labels):
        pass

    @abstractmethod
    def predict(self, test_data):
        pass

    def train_test_split(self, X, y):
        return train_test_split(X, y, test_size=self._test_size, random_state=42)


class AudioCassifier(Classifier):
    def __int__(self):
        pass

    def convertAudioToNumpy(self, audio_data):
        tbd...



from sklearn.svm import SVC
class SVDClassifier(AudioCassifier):

    def __init__(self):
        self.__classifier = SVC(kernel='rbf')
    def train(self, training_data, labels):
        self.__classifier.fit(training_data, labels)


    def predict(self, test_data):
        y_pred = self.__classifier.predict(test_data)
        return y_pred


class NNClassifier(AudioCassifier):

    def train(self, training_data):
        pass


    def predict(self, test_data):
        pass



classifier = SVDClassifier()
classifier.test_size = 0.2