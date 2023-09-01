from pathlib import Path
import numpy as np
import pandas as pd
import librosa

class AudioLoader:
    def __init__(self):
        self._csv_path = Path("../data/csv/")
        
    def load_files_from_folder(self, folder_path:str, label = "unlabled") -> pd.DataFrame:
        data = []

        for file in folder_path.iterdir():
            if file.is_file():
                audio_file, sample_rate = librosa.load(file)
                data.append({'Audio': audio_file, 'label': label, "sample_rate": sample_rate})

        df = pd.DataFrame(data)
        return df


    
    def merge_dataframes(self, data_filepath1: str, data_filepath2: str):
        data_file1 = pd.read_csv(data_filepath1)
        data_file2 = pd.read_csv(data_filepath2)
        dataframe = pd.concat([data_file1, data_file2])
        dataframe['features'] = dataframe['features'].apply(lambda x: np.array(x.strip('[]').split()).astype(float))
        output_filepath = self._csv_path / "data.csv"
        dataframe.to_csv(output_filepath, index=False)

        
        return dataframe

    
