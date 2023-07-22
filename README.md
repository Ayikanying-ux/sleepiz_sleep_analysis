# Audio Snoring Detection using MFCC Features

This Python code provides a framework for audio snoring detection using Mel-frequency cepstral coefficients (MFCC) features and Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel. The code processes audio data related to snoring and non-snoring sounds, extracts MFCC features, visualizes the audio and features, and builds a classifier to predict whether an audio sample belongs to the "snoring" class or "non-snoring" class.

## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `librosa`, `glob`, `os`, `sklearn`, `IPython`

## Usage

1. Import the `AudioProcessing` class from the provided Python script.

2. Create an instance of the `AudioProcessing` class, providing the file paths for snoring and non-snoring audio data.

3. Call the `train_model` method to train the SVM classifier and visualize the confusion matrix.

4. The output of the `train_model` method includes the predicted labels (`y_pred`) and the confusion matrix (`cm`) for evaluating the classifier's performance.

## Example

```python
from AudioProcessing import AudioProcessing

# Replace these file paths with your own snoring and non-snoring audio datasets
snoring_audio_path = "path/to/snoring_audio"
non_snoring_audio_path = "path/to/non_snoring_audio"

# Create an instance of the AudioProcessing class
audio_processor = AudioProcessing(snoring_audio_path, non_snoring_audio_path)

# Visualize and process snoring audio
audio_processor.snoring_visualization()

# Visualize and process non-snoring audio
audio_processor.non_snoring_visualization()

# Train the SVM classifier and display the confusion matrix
y_pred, confusion_matrix = audio_processor.train_model()
print("Predicted Labels:", y_pred)
print("Confusion Matrix:")
print(confusion_matrix)