from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from ..FeatureExtraction.extract_features import extract_mfcc_features

class AudioLoader:
    def __init__(self):
        self._csv_path = Path("../data/csv/")
        self._extract_feature = extract_mfcc_features
        
    def load_files_to_dataframe(self, folder_path:str):
        data = []

        for file in folder_path.iterdir():
            if file.is_file():
                audio_file, sample_rate = librosa.load(file)
                label = int(file.parent.name[-1])  # Extract label from parent folder name
                class_name = 'snoring' if label == 1 else 'non_snoring'
                data.append({'Audio': audio_file, 'label': class_name, "sample_rate": sample_rate})

        df = pd.DataFrame(data)

        #df.to_csv(self._csv_path / f"{data[0]['label']}.csv", index=False)
        return df

    
    def merge_dataframes(self, data_filepath1: str, data_filepath2: str):
        data_file1 = pd.read_csv(data_filepath1)
        data_file2 = pd.read_csv(data_filepath2)
        dataframe = pd.concat([data_file1, data_file2])
        dataframe['features'] = dataframe['features'].apply(lambda x: np.array(x.strip('[]').split()).astype(float))
        output_filepath = self._csv_path / "data.csv"
        dataframe.to_csv(output_filepath, index=False)

        
        return dataframe

    
