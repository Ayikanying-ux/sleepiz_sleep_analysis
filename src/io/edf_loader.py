from src.io.data_loader import AudioLoader
import pyedflib
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import librosa


class PSGLoader(AudioLoader):
    def __init__(self, channel = 'Mic', target_audio_length = 1, namespace = {'ns': 'http://www.respironics.com/PatientStudy.xsd'}, rml_file_nbr =2,  **kwargs):
        super().__init__(**kwargs)
        self.__channel = channel
        self.__target_audio_length = target_audio_length
        self.__namespace = namespace

    def load_audio_file(self, edf_file_name: str, rml_file_name: str, channel='mic', rml_file_nbr = 2):
        # The rml_file_nbr variable is only needed for this specific dataset. The PSG files have one rml file for each session, but 4 edf files, each one hour long.
        snoring_label_df = self.__read_snoring_labels(rml_file_name)
        data, sample_rate = self.__read_audio_file(edf_file_name)
        split_labels = self.__split_snoring_labels(snoring_label_df, rml_file_nbr=rml_file_nbr)

        self._add_audio_signal(data, sample_rate, split_labels)

    def __read_snoring_labels(self, rml_file_name: str):
        tree = ET.parse(rml_file_name)
        root = tree.getroot()



        scoring_data = root.find('.//ns:ScoringData', self.__namespace)
        start_times = []
        durations = []

        if scoring_data is not None:
            # Iterate over all Event elements under ScoringData
            for event in scoring_data.findall('.//ns:Event', self.__namespace):
                event_type = event.attrib.get('Type')
                if event_type == 'Snore':
                    start = event.attrib.get('Start')
                    duration = event.attrib.get('Duration')

                    # Append these to our lists
                    start_times.append(float(start))
                    durations.append(float(duration))
        else:
            print('ScoringData node not found.')

        # Create a DataFrame
        snoring_label_df = pd.DataFrame({
            'Start': start_times,
            'Duration': durations
        })
        return snoring_label_df

    def __read_audio_file(self, edf_file_name: str):
        edf_data = pyedflib.EdfReader(edf_file_name)
        # Get the total number of signals in the file
        n = edf_data.signals_in_file

        # Get the labels for all channels
        signal_labels = edf_data.getSignalLabels()

        # Initialize dictionaries to hold the signal data and sample rates
        data = {}

        # Loop through all channels to find the "Mic" and "Tracheal" channels
        for i in range(n):
            label = signal_labels[i]
            if label == self.__channel:
                header = edf_data.getSignalHeader(i)
                data["signal"] = edf_data.readSignal(i)
                data["sample_rate"] = header["sample_rate"]

        # Close the EDF file
        edf_data.close()
        return data["signal"], data["sample_rate"]

    def __split_snoring_labels(self, snoring_label_df, rml_file_nbr=0):
        y = (rml_file_nbr - 1) * np.zeros(3600)
        snoring_label_df['Start'] = snoring_label_df['Start'] - 3600
        for index, row in snoring_label_df.iterrows():
            start = int(np.floor(row['Start'] / self.__target_audio_length))
            end = int(np.ceil((row['Start'] + row['Duration']) / self.__target_audio_length))
            y[start:end] = 1
        return y

if __name__ == "__main__":
    file_name = "../../data/additional_files/00000995-100507[002].edf"
    rml_file_name = "../../data/additional_files/00000995-100507.rml"

    psg_loader = PSGLoader()
    psg_loader.load_audio_file(file_name, rml_file_name)

    edf_data = psg_loader.get_data()