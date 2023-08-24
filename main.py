from src.IO.DataLoader import DataLoader
from src.FeatureExtraction.MFCC import MFCCFeatureExtractor
from utils.model_trainer import ModelTrainer
from src.Visualization.visualization import VisualizeSnoring, VisualizeNonSnoring

if __name__ == "__main__":
    # File path for audio data
    snoring_audio_filepath = "data/1"
    non_snoring_audio_filepath = 'data/0'

    # Path to created csv files for both snoring and non snoring
    non_snoring_csv_filepath = "dataset/csv/non_snoring_data.csv"
    snoring_csv_filepath = "dataset/csv/snoring_data.csv"

    # Model filepath
    model_filepath = "model/classifier.pkl"
    '''
        data_loader - Create an instance of DataLoader
        df_snoring - Load snoring audio file to a pandas dataframe
        df_non_snoring - Load Non snoring audio file to pandas dataframe
        Then save them in csv format
    '''
    data_loader = DataLoader(snoring_audio_filepath, non_snoring_audio_filepath)
    df_snoring = data_loader.load_audio_data(snoring_audio_filepath, 1, "snoring")
    df_non_snoring = data_loader.load_audio_data(non_snoring_audio_filepath, 0, "non-snoring")
    df_snoring.to_csv(snoring_csv_filepath, index=False)
    df_non_snoring.to_csv(non_snoring_csv_filepath, index=False)

    feature_extractor = MFCCFeatureExtractor()
    # Perform visualizations (Snoring)
    snoring = VisualizeSnoring(feature_extractor, snoring_csv_filepath, snoring_audio_filepath="data/1")

    wave = snoring.waveplot() # Plot the wave form
    spectrum = snoring.plotPowerSpectrum() # Power Spectrum Plot
    spectogram = snoring.plotSpectogram() # Spectogrum Plot
    features = snoring.plotMFCC() # MFCC Plot


    # Perform visualizations (Non-snoring)
    non_snoring = VisualizeNonSnoring(feature_extractor, non_snoring_csv_filepath, non_snoring_audio_filepath="data/0")
    wave = non_snoring.waveplot()# Plot the wave form
    spectrum = non_snoring.plotPowerSpectrum()# Power Spectrum Plot
    spectogram = non_snoring.plotSpectogram()# Spectogrum Plot
    features = non_snoring.plotMFCC()# MFCC Plot

    # Model Building
    model = ModelTrainer(snoring_csv_filepath, non_snoring_csv_filepath, model_filepath)

    # Merge dataframes and train the model
    model.merge_dataframe()

    # Train model
    model.train_model()

    # Save model
    model.save_model()