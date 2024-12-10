import os
import shutil
import torch
import torch.nn as nn
import librosa
import numpy as np

# Model definition should match the training script exactly
class DogSoundModel(nn.Module):
    def __init__(self):
        super(DogSoundModel, self).__init__()
        self.fc1 = nn.Linear(13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)  # Additional hidden layer
        self.fc4 = nn.Linear(32, 16)  # Additional hidden layer
        self.fc5 = nn.Linear(16, 3)   # Adjusted output layer
        self.dropout = nn.Dropout(p=0.5)  # Dropout for regularization

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# Load the model with matching architecture
model = DogSoundModel()
model.load_state_dict(torch.load('dog_sound_model.pth', weights_only=True))
model.eval()

# Function to extract features from audio
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Classify files and move them to the corresponding folders in the dataset
def classify_files(model, test_data_folder, dataset_folder):
    model.eval()
    classifications = {'bark': 0, 'whine': 1, 'miscellaneous': 2}
    reversed_classifications = {v: k for k, v in classifications.items()}

    for filename in os.listdir(test_data_folder):
        if filename.endswith('.wav'):
            file_path = os.path.join(test_data_folder, filename)
            features = extract_features(file_path)
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                predicted_label = reversed_classifications[predicted.item()]

                # Move the file to the corresponding folder in the dataset
                target_folder = os.path.join(dataset_folder, predicted_label)
                os.makedirs(target_folder, exist_ok=True)
                shutil.move(file_path, os.path.join(target_folder, filename))
                print(f'Moved {filename} to {predicted_label} folder in dataset.')

# Path to your test data and dataset
test_data_folder = 'dog_sound_recognition/dataset/test_data'
dataset_folder = 'dog_sound_recognition/dataset'

# Classify the files
classify_files(model, test_data_folder, dataset_folder)
