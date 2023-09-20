import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.collections import PatchCollection

class ModelFitVisualizer:


    def create_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.close(fig)
        return fig

    def plot_audio_with_labels(self, X, sample_rates, y_true = None, y_pred = None, target_sample_rate=22050):
        # Check if lengths of X, y, and sample_rates are the same
        if (len(X) != len(sample_rates)):
            raise ValueError("Lengths of X, y, and sample_rates do not match")
        if (y_true is not None and len(X) != len(y_true)):
            raise ValueError("Lengths of X, y, and sample_rates do not match")
        if (y_pred is not None and len(X) != len(y_pred)):
            raise ValueError("Lengths of X, y, and sample_rates do not match")

        # Resample each audio signal to the target sample rate and then concatenate
        if len(np.unique(sample_rates))>1:
            X = [librosa.resample(x, orig_sr=src, target_sr=target_sample_rate) for x, src in
                       zip(X, sample_rates)]

        concatenated_audio = np.concatenate(X)

        # Compute starting positions of each audio sample in the concatenated audio in seconds
        durations = [len(x) / target_sample_rate for x in X]
        start_times = np.cumsum([0] + durations[:-1])

        plt.figure(figsize=(15, 5))

        # Display the waveform
        librosa.display.waveshow(concatenated_audio, sr=target_sample_rate)
        legend_elements = [Line2D([0], [0], color='b', lw=2, label='Audio Waveform')]

        min_val_audio = min(concatenated_audio)
        max_val_audio = max(concatenated_audio)

        if (y_true is not None):
            rectangles = []
            positive_indices = np.where(y_true == 1)[0]
            for idx in positive_indices:
                start_time = start_times[idx]
                duration = durations[idx]
                rect = patches.Rectangle((start_time, min_val_audio), duration,
                                         max_val_audio - min_val_audio)
                rectangles.append(rect)

            p = PatchCollection(rectangles, facecolor='red', alpha=0.5)
            plt.gca().add_collection(p)


            legend_elements.append(patches.Patch(facecolor="red", edgecolor="red", alpha=0.5, label='True Snoring'))

        if(y_pred is not None):
            rectangles = []
            positive_indices = np.where(y_pred == 1)[0]
            for idx in positive_indices:
                start_time = start_times[idx]
                duration = durations[idx]
                rect = patches.Rectangle((start_time, min_val_audio), duration,
                                         max_val_audio - min_val_audio)
                rectangles.append(rect)

            p = PatchCollection(rectangles, facecolor='green', alpha=0.5)
            plt.gca().add_collection(p)
            legend_elements.append(patches.Patch(facecolor="green", edgecolor="red", alpha=0.5, label='Predicted Snoring'))



        plt.legend(handles=legend_elements, loc='upper right')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Audio Waveform with Highlighted Snoring Segments')
        plt.show()



# Profiling code
if __name__ == "__main__":
    from src.model.svm import SVModel
    from src.io.edf_loader import PSGLoader
    from src.audiopreprocessing.audio_preprocessor import AudioPreprocessor

    file_name = "../../data/additional_files/00000995-100507[002].edf"
    rml_file_name = "../../data/additional_files/00000995-100507.rml"


    psg_loader = PSGLoader(channel="Mic")
    psg_loader.load_audio_file(file_name, rml_file_name)

    edf_data = psg_loader.get_data()

    X_mic = np.array(edf_data[psg_loader.AUDIO_COL_NAME])
    X_mic_sample_rate = edf_data[psg_loader.SAMPLE_RATE_COL_NAME]
    Y_mic = edf_data[psg_loader.LABEL_COL_NAME]

    preprocessor = AudioPreprocessor(X_mic, X_mic_sample_rate)
    X_mic_mfcc = preprocessor.extract_mffc_features()
    model_2 = SVModel(C=5, kernel="rbf")
    model_2.train_model(X_mic_mfcc, Y_mic)

    Y_mic_pred = model_2.classify(X_mic_mfcc)

    start_index = 0
    visualization_length = 600
    end_index = start_index + visualization_length

    model_visualizer = ModelFitVisualizer()
    model_visualizer.plot_audio_with_labels(X_mic[start_index: end_index], Y_mic[start_index: end_index],
                                            X_mic_sample_rate[start_index: end_index],
                                            y_pred=Y_mic_pred[start_index: end_index])