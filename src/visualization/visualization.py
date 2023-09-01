import librosa
import numpy as np
import matplotlib.pyplot as plt


class AudioVisualizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def CreateWavePlot(self, column_name: np.array, data_nbr: int = 0, width=10, height=4, title=None):
        audio = self.dataframe[column_name][data_nbr]
        fig = plt.figure(figsize=(width, height))
        librosa.display.waveshow(audio, alpha=0.25)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        if title:
            plt.title(title)
        return fig
    
    def CreatePowerSpectrumPlot(self, audio_column: np.array, sample_rate_column, data_nbr: int = 0, width=10, height=4, title=None):
        
        spectrum = np.abs(np.fft.fft(self.dataframe[audio_column][data_nbr]))
        frequencies = np.fft.fftfreq(len(self.dataframe[audio_column][data_nbr]), d=1/self.dataframe[sample_rate_column][data_nbr])
        
        fig = plt.figure(figsize=(width, height))
        plt.plot(frequencies, spectrum)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        if title:
            plt.title(title)
        return fig

    def CreateMFCCplot(self, features_column, width: int = 10, height: int = 4, title: str = None):
        MFFCfeatures = self.dataframe[features_column].to_list()
        features = np.array(MFFCfeatures)
        fig = plt.figure(figsize=(width, height))
        librosa.display.specshow(features.T, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        if title:
            plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        return fig

    def CreateSpectogram(self, audio_column: np.array, sample_rate_column, data_nbr: int = 0, width: int = 10, height: int = 4, title=None):

        fig = plt.figure(figsize=(width, height))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(self.dataframe[audio_column][0])), ref=np.max)
        librosa.display.specshow(spectrogram, sr=self.dataframe[sample_rate_column][0], x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        if title:
            plt.title('Spectrogram of Snoring Audio')
        return fig