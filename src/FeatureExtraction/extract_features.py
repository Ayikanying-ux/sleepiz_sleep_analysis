import numpy as np
import librosa

class ExtractMFCCFeatures:
    def __init__(self, num_mfcc=20, num_filter_banks=26):
        self.num_mfcc = num_mfcc
        self.num_filter_banks = num_filter_banks

    def extract_mfcc_features(self, audio_data : np.array, sample_rate):
        filter_banks = librosa.filters.mel(sr=sample_rate, n_fft=2048, n_mels=self.num_filter_banks)
        filter_banks_features = np.dot(filter_banks, np.abs(librosa.stft(audio_data))**2.0)

        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(filter_banks_features), n_mfcc=self.num_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_features = np.concatenate((mfcc_mean, mfcc_std))
        return mfcc_features