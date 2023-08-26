import librosa
import numpy as np
import matplotlib.pyplot as plt


class AudioVisualization:
    def __init__(self, audio):
        self.audio = audio

    def waveplot(self, data_nbr: int = 0, width=10, height=4, title=None):
        audio, _ = librosa.load(self.audio[data_nbr])
        print(audio)
        fig = plt.figure(figsize=(width, height))
        librosa.display.waveshow(audio, alpha=0.25)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        if title:
            plt.title(title)
        return fig
    
    def powerSpectrumPlot(self, data_nbr: int = 0, width=10, height=4, title=None):
        audio, sample_rate = librosa.load(self.audio[data_nbr])
        if sample_rate is None:
            raise ValueError("Sample rate must be provided for power spectrum plot")
        
        spectrum = np.abs(np.fft.fft(audio))
        frequencies = np.fft.fftfreq(len(audio), d=1/sample_rate)
        
        fig = plt.figure(figsize=(width, height))
        plt.plot(frequencies, spectrum)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        if title:
            plt.title(title)
        return fig

    def MFCCplot(self):
        pass

    def spectogram(self, data_nbr: int = 0, width: int = 10, height: int = 4, title=None):
        audio, sample_rate = librosa.load(self.audio[data_nbr])

        fig = plt.figure(figsize=(width, height))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        if title:
            plt.title('Spectrogram of Snoring Audio')
        return fig