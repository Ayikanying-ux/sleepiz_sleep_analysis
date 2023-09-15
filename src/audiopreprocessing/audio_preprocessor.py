import numpy as np
import librosa

class AudioPreprocessor:
    def __init__(self, audio_signals: np.array, sample_rates):
        self.__audio_signals = audio_signals
        self.__sample_rates = sample_rates

    def resample_audio(self, target_sample_rate):
        resampled_audio_signals = []
        new_sample_rates = []
        for signal_nbr, audio_signal in enumerate(self.__audio_signals):
            resampled_audio = librosa.resample(audio_signal, orig_sr=self.__sample_rates[signal_nbr], target_sr=target_sample_rate)
            resampled_audio = librosa.util.normalize(resampled_audio)
            resampled_audio_signals.append(resampled_audio)
            new_sample_rates.append(target_sample_rate)
        return np.array(resampled_audio_signals), new_sample_rates
    

    def extract_mffc_features(self):
        feature_list = []
        for index, audio_signal in enumerate(self.__audio_signals):
            sample_rate = self.__sample_rates[index]
            features = librosa.feature.mfcc(y=audio_signal, sr=sample_rate)
            mfcc_mean = np.mean(features, axis=1)
            feature_list.append(mfcc_mean)
        return np.array(feature_list)
