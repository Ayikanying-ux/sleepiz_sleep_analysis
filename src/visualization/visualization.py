import librosa
import numpy as np
import matplotlib.pyplot as plt


class AudioVisualizer:
    def __init__(self,  audio_signals: np.array, sample_rates, plot_width = 10, plot_height = 4):
        self.__audio_signals = audio_signals
        self.__sample_rates = sample_rates
        self.__width = plot_width
        self.__height = plot_height

    def create_wav_plot(self, audio_signal_index = 0, title=None):
        if audio_signal_index > len(self.__audio_signals):
            print(f"Only {len(self.__audio_signals)} audio signals provided, but tried to plot audio signal with index {audio_signal_index}")

        audio_signal = self.__audio_signals[audio_signal_index]
        sample_rate = self.__sample_rates[audio_signal_index]
        fig = plt.figure(figsize=(self.__width, self.__height));
        librosa.display.waveshow(audio_signal, sr=sample_rate, alpha=0.25)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        if title:
            plt.title(title)
        else:
            plt.title(f"Wave plot of audio signal number {audio_signal_index}")
        plt.close(fig)
        return fig
    
    def create_frequency_plot(self, audio_signal_index = 0, title=None):
        audio_signal = self.__audio_signals[audio_signal_index]
        sample_rate = self.__sample_rates[audio_signal_index]

        fft_output = np.fft.fft(audio_signal)
        magnitude = np.abs(fft_output)[:len(audio_signal) // 2]
        frequency = np.fft.fftfreq(len(audio_signal), 1 / sample_rate)[:len(audio_signal) // 2]

        # Logarithmically spaced frequency bins
        num_bins = 50  # You can adjust this as needed
        log_freq = np.logspace(np.log2(frequency[1]), np.log2(frequency[-1]), num_bins)
        log_magnitude = np.histogram(frequency, bins=log_freq, weights=magnitude)[0]

        # Create bar plot
        fig = plt.figure(figsize=(self.__width, self.__height))
        plt.bar(log_freq[:-1], log_magnitude, width=np.diff(log_freq))
        plt.xscale('log')

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        if title:
            plt.title(title)
        else:
            plt.title(f"Fourier Transform plot of audio signal number {audio_signal_index}")

        plt.close(fig)
        return fig

    def create_mfcc_plot_whole_dataset(self, title: str = "MFCC of entire dataset"):
        mfcc_features = [np.mean(librosa.feature.mfcc(y=self.__audio_signals[i], sr=self.__sample_rates[i]), axis=1) for i in len(self.__audio_signals)]
        features = np.array(mfcc_features)
        fig = plt.figure(figsize=(self.__width, self.__height))
        librosa.display.specshow(features.T, x_axis='time')
        plt.colorbar(format='%+2.0f dB')

        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.close(fig)
        return fig

    def create_spectrogram_plot(self, audio_signal_index = 0, title=None):

        fig = plt.figure(figsize=(self.__width, self.__height))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(self.__audio_signals[audio_signal_index])), ref=np.max)
        librosa.display.specshow(spectrogram, sr=self.__sample_rates[audio_signal_index], x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')

        if title:
            plt.title(title)
        else:
            plt.title(f"Spectogram of audio signal number {audio_signal_index}")
        plt.close(fig)
        return fig

    def set_plot_height_and_width(self, width: int, height: int):
        self.__width = width
        self.__height = height