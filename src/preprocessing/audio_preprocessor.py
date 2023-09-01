import numpy as np
import pandas as pd
import librosa
# from src.FeatureExtraction.extract_features import extrac/t_mfcc_features


class AudioDFPreprocessor:
    def __init__(self,  audio_column_name:str = "Audio",sample_rate_column_name= "sample_rate"):
        self.__audio_column_name = audio_column_name
        self.__sample_rate_column_name = sample_rate_column_name

    def resample_audio(self, data: pd.DataFrame, target_sample_rate: int =22050 ):
        def resample_audio(row):
            return librosa.resample(row[self.__audio_column_name], orig_sr=row[self.__sample_rate_column_name], target_sr=target_sample_rate)
        data[self.__audio_column_name] = data.apply(resample_audio, axis=1)
        data[self.__sample_rate_column_name] = target_sample_rate

    

    def add_mffc_features(self, data: pd.DataFrame, num_mfcc: int = 20, num_filter_banks: int = 26):
        def extract_mfcc(row):
            audio_data = row[self.__audio_column_name]
            sample_rate = row[self.__sample_rate_column_name]
            filter_banks = librosa.filters.mel(sr=sample_rate, n_fft=2048, n_mels=num_filter_banks)
            filter_banks_features = np.dot(filter_banks, np.abs(librosa.stft(audio_data)) ** 2.0)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(filter_banks_features), n_mfcc=num_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_features = np.concatenate((mfcc_mean, mfcc_std))
            return mfcc_features

        data['mfccs'] = data.apply(extract_mfcc, axis=1)