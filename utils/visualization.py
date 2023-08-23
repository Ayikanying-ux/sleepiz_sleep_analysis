import os
import librosa
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
#matplotlib.use('TkAgg')



class VisualizeSnoring:
    def __init__(self, feature_extractor, snoring_csv_filepath, snoring_audio_filepath) -> None:
        self.feature_extractor = feature_extractor
        self.snoring_data_path = snoring_csv_filepath
        self.snoring_audio_filepath = snoring_audio_filepath
    
    def extractFeatures(self):
        # Load snoring audio data and extract MFCC features
        audio_filepath_snoring = self.snoring_audio_filepath
        snoring_data = pd.read_csv(self.snoring_data_path)
        audio_data_snoring = []
        sample_rate_snoring = []
        features_snoring=[]

        for index, row in snoring_data.iterrows():
            audio_file = row['audio']
            audio_file_path = os.path.join(audio_filepath_snoring, audio_file)
            audio_data, sample_rate = librosa.load(audio_file_path)
            audio_data_snoring.append(audio_data)
            sample_rate_snoring.append(sample_rate)
            
            # Extract MFCC features
            mfcc_features = self.feature_extractor.extract_mfcc_features(audio_data=audio_data, sample_rate=sample_rate)
            features_snoring.append(mfcc_features)

        snoring_data['Audio_Data'] = audio_data_snoring
        snoring_data['Sample_Rate'] = sample_rate_snoring
        snoring_data['Features'] = features_snoring

        snoring_data.to_csv('dataset/csv/snoring_data.csv', index=False)
        return audio_data_snoring, sample_rate_snoring, features_snoring

    def waveplot(self):

        audio_data_snoring, sample_rate_snoring, _ = self.extractFeatures()

        # Display audio
        display(Audio(audio_data_snoring[0], rate=sample_rate_snoring[0]))

        # Wave plot for snoring
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(audio_data_snoring[0], alpha=0.25)
        plt.title("Wave Plot for Snoring")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    def plotPowerSpectrum(self):
        audio_data_snoring, sample_rate_snoring, _ = self.extractFeatures()
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

    def plotSpectogram(self):
        audio_data_snoring, sample_rate_snoring, _ = self.extractFeatures()
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data_snoring[0])), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sample_rate_snoring[0], x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram of Snoring Audio')
        plt.show()

        # Plot MFFC

    def plotMFCC(self):
        _, _,features_snoring = self.extractFeatures()
        features_snoring = np.array(features_snoring)
        #print(list(features_snoring))
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(features_snoring.T, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Features')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.show()



class VisualizeNonSnoring:
    def __init__(self, feature_extractor, non_snoring_csv_filepath, non_snoring_audio_filepath) -> None:
        self.feature_extractor = feature_extractor
        self.non_snoring_csv_filepath = non_snoring_csv_filepath
        self.non_snoring_audio_filepath = non_snoring_audio_filepath
    
    def extractFeatures(self):
        non_audio_filepath_snoring = self.non_snoring_audio_filepath
        non_snoring_data = pd.read_csv(self.non_snoring_csv_filepath)
        non_snoring_audio_data = []
        non_snoring_sample_rate = []
        non_snoring_features = []

        for index, row in non_snoring_data.iterrows():
            audio_file = row['audio']
            audio_file_path = os.path.join(non_audio_filepath_snoring, audio_file)
            audio_data, sample_rate = librosa.load(audio_file_path)
            non_snoring_audio_data.append(audio_data)
            non_snoring_sample_rate.append(sample_rate)
            
            # Extract MFCC features
            mfcc_features = self.feature_extractor.extract_mfcc_features(audio_data, sample_rate)
            non_snoring_features.append(mfcc_features)

        non_snoring_data['Audio_Data'] = non_snoring_audio_data
        non_snoring_data['Sample_Rate'] = non_snoring_sample_rate
        non_snoring_data['Features'] = non_snoring_features
        
        non_snoring_data.to_csv('dataset/csv/non_snoring_data.csv', index=False)
        return non_snoring_audio_data, non_snoring_sample_rate, non_snoring_features

    def waveplot(self):

        non_snoring_audio_data, non_snoring_sample_rate, _ = self.extractFeatures()

        # Display audio
        display(Audio(non_snoring_audio_data[0], rate=non_snoring_sample_rate[0]))

        # Wave plot for snoring
        plt.figure(figsize=(10, 4))
        librosa.display.waveshow(non_snoring_audio_data[0], alpha=0.25)
        plt.title("Wave Plot for Snoring")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()

    def plotPowerSpectrum(self):
        non_snoring_audio_data, non_snoring_sample_rate, _ = self.extractFeatures()
        # Plot the FFT
        plt.figure(figsize=(8, 4))
        fft = np.fft.fft(non_snoring_audio_data[0])
        magnitude = np.abs(fft)
        frequency = np.linspace(0, non_snoring_sample_rate, len(magnitude))
        #half_spectrum = fft[:int(len(fft) / 2)]
        #half_frequency = frequency[:int(len(fft)/2)]
        plt.plot(frequency[:len(frequency) // 2], magnitude[:len(frequency) // 2])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT of Snoring Audio')
        plt.show()

    def plotSpectogram(self):
        non_snoring_audio_data, sample_rate_snoring, _ = self.extractFeatures()
        # Plot the spectrogram
        plt.figure(figsize=(10, 4))
        spectrogram = librosa.amplitude_to_db(np.abs(librosa.stft(non_snoring_audio_data[0])), ref=np.max)
        librosa.display.specshow(spectrogram, sr=sample_rate_snoring[0], x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram of Snoring Audio')
        plt.show()

        # Plot MFFC

    def plotMFCC(self):
        _, _, non_snoring_features = self.extractFeatures()
        print(non_snoring_features)
        non_snoring_features = np.array(non_snoring_features)
        #print(list(features_snoring))
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(non_snoring_features.T, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC Features')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficients')
        plt.show()