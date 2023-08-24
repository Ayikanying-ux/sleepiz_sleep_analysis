
from pathlib import Path
import librosa

class AudioLoader():
    def __init__(self,data_path: str):
        self.data_path = Path(data_path)

    def readAudiofilesInFolder(self, subfolder = ".", file_ending = "wav"):
        audio_file_names = (self.data_path / subfolder).glob('*.' + '.file_ending')
        audio_data = [librosa.load(file_name) for file_name in audio_file_names]
        return audio_data