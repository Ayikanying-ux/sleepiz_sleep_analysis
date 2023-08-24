from pathlib import Path
import glob
import os
import pandas as pd
import librosa

class DataLoader:
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def get_data_as_dataframe(self):
        data = []

        for file in self.folder_path.iterdir():
            if file.is_file():
                label = int(file.parent.name[-1])  # Extract label from parent folder name
                class_name = 'snoring' if label == 1 else 'non_snoring'
                data.append({'filename': file.name, 'id': label, 'class': class_name})

        df = pd.DataFrame(data)
        return df

    
