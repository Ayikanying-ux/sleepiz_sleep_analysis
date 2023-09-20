from pathlib import Path
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import pandas as pd
import librosa

class AudioLoader(ABC):
    AUDIO_COL_NAME = 'audio'
    LABEL_COL_NAME = 'label'
    SAMPLE_RATE_COL_NAME = 'sample_rate'
    def __init__(self, audio_segment_length:float = 1):
        self._data = pd.DataFrame(columns=[self.AUDIO_COL_NAME, self.LABEL_COL_NAME, self.SAMPLE_RATE_COL_NAME])
        self._segment_length = audio_segment_length

    def get_data(self):
        if self._data.empty:
            print('No data loaded, please load data first')
        return self._data

    def _split_audio_file(self, audio_signal: np.array, sample_rate: int) -> List[np.array]:
        frame_length = int(sample_rate * self._segment_length)
        hop_length = frame_length
        split_signal = []
        for start_idx in range(0, len(audio_signal), hop_length):
            end_idx = int(start_idx + frame_length)
            if(end_idx > len(audio_signal)):
                return
            split_signal.append(audio_signal[start_idx:end_idx])
        return split_signal

    def _add_audio_signal(self, audio_signal: np.array, sample_rate: int, label):
        signal_lengt = len(audio_signal) / sample_rate
        if signal_lengt <= self._segment_length:
            if signal_lengt < self._segment_length:
                print("Audio file is shorter than ", self._segment_length, " but will be added to the dataset none-the-less")
            new_row = pd.Series({
                self.AUDIO_COL_NAME: audio_signal,
                self.LABEL_COL_NAME: label,
                self.SAMPLE_RATE_COL_NAME: sample_rate
            })
            self._data = pd.concat([self._data, new_row.to_frame().T], ignore_index=True)
        else:
            signals = self._split_audio_file(audio_signal, sample_rate)
            num_signals = len(signals)
            sample_rates = [sample_rate] * num_signals
            if len(label) == 1:
                label = [label] * num_signals
            new_rows = pd.DataFrame({self.AUDIO_COL_NAME: signals,
                                     self.LABEL_COL_NAME: label,
                                     self.SAMPLE_RATE_COL_NAME: sample_rates})
            self._data = pd.concat([self._data, new_rows], ignore_index=True)



# For adding multiple rows (another DataFrame)





class AudioFileLoader(AudioLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def load_audio_files_from_folder(self, folder_path: str, label, file_format='wav'):
        for file_path in Path(folder_path).iterdir():
            if file_path.suffix == f".{file_format}":
                self.load_audio_file(file_path, label, file_format=file_format)

    def load_audio_file(self, path_to_file: str, label, file_format: str = 'wav'):
        if path_to_file.is_file() and path_to_file.suffix == f".{file_format}":
            try:
                audio_signal, sample_rate = librosa.load(path_to_file, sr=None)
                self._add_audio_signal(audio_signal, sample_rate, label)
            except Exception as e:
                print(f"An error occurred while reading the file {path_to_file}: {e}")
        else:
            print(f"The file {path_to_file} doesn't exist or is not a {file_format} file.")
