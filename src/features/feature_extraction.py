import numpy as np
import librosa


def extract_mfcc_features(audio_data: np.array, 
                          sample_rate: int, num_mfcc: int = 20, 
                          num_filter_banks: int = 26) -> np.array:
    filter_banks = librosa.filters.mel(sr=sample_rate, n_fft=2048, n_mels=num_filter_banks)
    filter_banks_features = np.dot(filter_banks, np.abs(librosa.stft(audio_data)) ** 2.0)

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(filter_banks_features), n_mfcc=num_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_features = np.concatenate((mfcc_mean, mfcc_std))
    return mfcc_features