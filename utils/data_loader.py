import os
import glob
import pandas as pd
import librosa

class DataLoader:
    def __init__(self, audio_filepath_snoring, audio_filepath_non_snoring):
        self.audio_filepath_snoring = audio_filepath_snoring
        self.audio_filepath_non_snoring = audio_filepath_non_snoring

    def load_audio_data(self, audio_filepath, class_id, label):
        file_pattern = '*.wav'
        file_names = []
        labels = []
        file_paths = glob.glob(os.path.join(audio_filepath, '**', file_pattern), recursive=True)

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            file_names.append(file_name)
            labels.append(label)

        data = {'audio': file_names, 'Labels': labels, "class_id": class_id}
        df = pd.DataFrame(data)

        return df
    
