from utils.data_loader import DataLoader
from utils.feature_extraction import FeatureExtractor
from utils.model_trainer import ModelTrainer
from utils.visualization import Visualization

if __name__ == "__main__":
    # Provide the file paths for snoring and non-snoring audio data
    audio_filepath_snoring = "data/1"
    audio_filepath_non_snoring = 'data/0'

    # Create instances of modules
    data_loader = DataLoader(audio_filepath_snoring, audio_filepath_non_snoring)
    df_snoring = data_loader.load_audio_data(audio_filepath_snoring, 1, "snoring")
    df_non_snoring = data_loader.load_audio_data(audio_filepath_non_snoring, 0, "non-snoring")

    df_snoring.to_csv('dataset/csv/snoring_data.csv', index=False)
    df_non_snoring.to_csv('dataset/csv/non_snoring_data.csv', index=False)

    feature_extractor = FeatureExtractor()
    visualization = Visualization(feature_extractor)
    model_trainer = ModelTrainer(data_loader, feature_extractor)

    # Perform snoring visualization and save the dataframe
    df_snoring = visualization.snoring_visualization()

    # Perform non-snoring visualization and save the dataframe
    df_non_snoring = visualization.non_snoring_visualization()

    # Merge dataframes and train the model
    model_trainer.train_model()