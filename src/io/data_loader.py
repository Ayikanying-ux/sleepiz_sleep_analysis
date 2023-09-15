from pathlib import Path
import numpy as np
import pandas as pd
import librosa

class AudioLoader:
    AUDIO_COL_NAME = 'audio'
    LABEL_COL_NAME = 'label'
    SAMPLE_RATE_COL_NAME = 'sample_rate'
    def __init__(self):
        self.__data = pd.DataFrame(columns=[self.AUDIO_COL_NAME, self.LABEL_COL_NAME, self.SAMPLE_RATE_COL_NAME])

    def load_audio_files_from_folder(self, folder_path:str, label, file_format = 'wav'):
        for file_path in Path(folder_path).iterdir():
            if file_path.suffix == f".{file_format}":
                self.load_audio_file(file_path, label, file_format = file_format)

    def load_audio_file(self, path_to_file:str, label,  file_format: str = 'wav'):
        if path_to_file.is_file() and path_to_file.suffix == f".{file_format}":
            try:
                audio_signal, sample_rate = librosa.load(path_to_file, sr=None)
                self.__data.loc[len(self.__data)] = {self.AUDIO_COL_NAME: np.array(audio_signal),
                                                     self.LABEL_COL_NAME: label,
                                                     self.SAMPLE_RATE_COL_NAME: sample_rate}
            except Exception as e:
                print(f"An error occurred while reading the file {path_to_file}: {e}")
        else:
            print(f"The file {path_to_file} doesn't exist or is not a {file_format} file.")

    def get_data(self):
        if self.__data.empty:
            print('No data loaded, please load data first')
        return self.__data


    
