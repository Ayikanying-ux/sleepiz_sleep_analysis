import librosa
import numpy as np
import pandas as pd
import librosa
from src.FeatureExtraction.extract_features import extract_mfcc_features


class AudioPreprocessor:
    def __init__(self, dataframe, target_sample_rate=22050, noise_factor=0.005):
        self.target_sample_rate = target_sample_rate
        self.noise_factor = noise_factor
        self.dataframe = dataframe
        self._feature_extractor = extract_mfcc_features

    def resample_audio_and_sample_rate(self, audio_column_name:str, sample_rate_column_name):
        for index, row in self.dataframe.iterrows():
            sample_rate = row[sample_rate_column_name]
            resampled_audio = librosa.resample(row[audio_column_name], orig_sr=sample_rate, target_sr=self.target_sample_rate)
            self.dataframe.at[index, audio_column_name] = resampled_audio
        return self.dataframe
    

    def extract_mffc_features(self, audio_column: np.array):
        self.dataframe['features'] = None
        for index, row in self.dataframe.iterrows():
            audio = row[audio_column]
            sample_rate = self.target_sample_rate
            features = self._feature_extractor(audio_data=audio, sample_rate=sample_rate)
            self.dataframe.at[index, "features"] = features
        return self.dataframe
    
    def preprocess(self, audio_column_name: np.ndarray, sample_rate_column_name: int):
        self.resample_audio_and_sample_rate(audio_column_name, sample_rate_column_name)
        extracted_features = self.extract_mffc_features(audio_column_name)
        return extracted_features