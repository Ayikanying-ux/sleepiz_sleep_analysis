import os
import librosa
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import numpy as np
#matplotlib.use('TkAgg')


class AudioVisualizer():

    def __init__(self, audio_data):
        self.audio_data = audio_data

    def createWavPlot(self, data_nbr: int = 0, width = 10, height = 5, title = None):
        fig = plt.figure(figsize=(width, height))
        librosa.display.waveshow(self.audio_data[data_nbr][0], alpha=0.25)
        if title:
            plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        return fig


