from pathlib import Path
import pandas as pd

class DataLoader:
    def __init__(self, folder_path, filename: str):
        self.folder_path = Path(folder_path)
        self.filename = filename
        self._csv_path = "../data/"
    def load_files_to_dataframe(self):
        data = []

        for file in self.folder_path.iterdir():
            if file.is_file():
                label = int(file.parent.name[-1])  # Extract label from parent folder name
                class_name = 'snoring' if label == 1 else 'non_snoring'
                data.append({'filename': file.name, 'class': label, 'label': class_name})

        df = pd.DataFrame(data)
        df.to_csv(self._csv_path + self.filename + ".csv")
        return df

    
