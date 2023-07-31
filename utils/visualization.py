import os
import librosa
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
#matplotlib.use('TkAgg')

from .data_loader import DataLoader
from .feature_extraction import FeatureExtractor

class Visualization:
    def __init__(self, feature_extractor) -> None:
        self.feature_extractor = feature_extractor

    def snoring_visualization(self):
        df_snoring = pd.read_csv("dataset/csv/snoring_data.csv")
        audio_filepath_snoring = 'data/1/'
        audio_data_snoring = []
        sample_rate_snoring = []
        features_snoring=[]

        for index, row in df_snoring.iterrows():
            audio_file = row['audio']
            audio_file_path = os.path.join(audio_filepath_snoring, audio_file)
            audio_data, sample_rate = librosa.load(audio_file_path)
            audio_data_snoring.append(audio_data)
            sample_rate_snoring.append(sample_rate)
            
            # Extract MFCC features
            mfcc_features = self.feature_extractor.extract_mfcc_features(audio_data, sample_rate)
            features_snoring.append(mfcc_features)

        df_snoring['Audio_Data'] = audio_data_snoring
        df_snoring['Sample_Rate'] = sample_rate_snoring
        df_snoring['Features'] = features_snoring

        # Display audio
        display(Audio(audio_data_snoring[0], rate=sample_rate_snoring[0]))

        # Wave plot for snoring
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data_snoring[0], alpha=0.25)
        plt.title("Wave Plot for Snoring")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()


        # Plot the FFT
        plt.figure(figsize=(8, 4))
        fft = np.fft.fft(audio_data_snoring[0])
        magnitude = np.abs(fft)
        frequency = np.linspace(0, sample_rate_snoring, len(magnitude))
        #half_spectrum = fft[:int(len(fft) / 2)]
        #half_frequency = frequency[:int(len(fft)/2)]
        plt.plot(frequency[:len(frequency) // 2], magnitude[:len(frequency) // 2])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT of Snoring Audio')
        plt.show()

        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data_snoring[0])), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sample_rate_snoring[0], x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram of Snoring Audio')
        plt.show()

        # Plot MFFC

        features_snoring = np.array(features_snoring)
        #print(list(features_snoring))
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(features_snoring.T, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Features')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.show()

        #df_snoring.to_csv('../dataset/csv/snoring_data.csv', index=False)

        return df_snoring
    
    def non_snoring_visualization(self):
        df_non_snoring = pd.read_csv("dataset/csv/non_snoring_data.csv")
        audio_filepath_non_snoring = 'data/0/'
        audio_data_non_snoring = []
        sample_rate_non_snoring = []
        features_non_snoring = []
        

        for index, row in df_non_snoring.iterrows():
            audio_file = row['audio']
            audio_file_path = os.path.join(audio_filepath_non_snoring, audio_file)
            audio_data, sample_rate = librosa.load(audio_file_path)
            audio_data_non_snoring.append(audio_data)
            sample_rate_non_snoring.append(sample_rate)

            # Extract MFCC features
            mfcc_features = self.feature_extractor.extract_mfcc_features(audio_data, sample_rate)
            features_non_snoring.append(mfcc_features)

        df_non_snoring['Audio_Data'] = audio_data_non_snoring
        df_non_snoring['Sample_Rate'] = sample_rate_non_snoring
        df_non_snoring['Features'] = features_non_snoring

        display(Audio(audio_data_non_snoring[0], rate=sample_rate_non_snoring[0]))

        # Waveform plot for non snoring
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data_non_snoring[99], alpha=0.25)
        plt.title("Wave plot for non-snoring")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

        # FFT for non-snoring Audio
        plt.figure(figsize=(8, 4))
        fft = np.fft.fft(audio_data_non_snoring[99])
        magnitude = np.abs(fft)
        frequency = np.linspace(0, sample_rate_non_snoring, len(magnitude))
        plt.plot(frequency[:len(frequency) // 2], magnitude[:len(frequency) // 2])

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT of Non Snoring Audio')
        plt.show()

        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data_non_snoring[99])), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sample_rate_non_snoring[0], x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram of Non-Snoring Audio')
        plt.show()

        features_non_snoring = np.array(features_non_snoring)

        # Plot MFCC
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(features_non_snoring.T, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Features')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.show()

        #df_non_snoring.to_csv('../dataset/csv/non_snoring_data.csv', index=False)

        return df_non_snoring
        