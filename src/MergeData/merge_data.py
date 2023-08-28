import pandas as pd

class MergeData:
    def __init__(self, data_filepath1, data_filepath, folderPath) -> None:
        self.data_filepath1 = data_filepath1
        self.data_filepath = data_filepath
        self.folderPath = folderPath

    def merge(self):
        dataframe = pd.concat(self.data_filepath1, self.data_filepath)
        dataframe.to_csv(self.folderPath + "data.csv")
        return dataframe