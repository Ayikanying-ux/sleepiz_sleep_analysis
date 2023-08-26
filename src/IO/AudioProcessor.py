from pathlib import Path

class AudioProcessor():
    def __init__(self, audio_path) -> None:
        self.audio_path = Path(audio_path)

    def read_audio_files_in_folder(self, subfolder='.', file_ending='wav'):
        audio_file_names = (self.audio_path / subfolder).glob('*.' + file_ending)
        audio_data = []
        for file_name in audio_file_names:
            audio_data.append(file_name)
        return audio_data