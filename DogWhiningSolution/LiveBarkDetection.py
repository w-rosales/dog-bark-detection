import os
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa

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
model.load_state_dict(torch.load('dog_sound_model.pth'))
model.eval()

# Function to extract features from audio
def extract_features(audio_data, sample_rate):
    n_fft = min(2048, len(audio_data))  # Adjust n_fft based on the length of the audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=n_fft)
    return np.mean(mfccs.T, axis=0)

# Function to process live audio for bark detection
def classify_audio(model, audio_data, sample_rate):
    features = extract_features(audio_data, sample_rate)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()

# Callback function to process audio in chunks
def audio_callback(indata, frames, time, status):
    sample_rate = 44100  # Sample rate of the microphone
    audio_data = np.squeeze(indata)
    if np.any(audio_data):  # Only process non-silent audio
        prediction = classify_audio(model, audio_data, sample_rate)
        if prediction == 0:
            print("Bark")
        elif prediction == 1:
            print("Whine")
        else:
            print("Miscellaneous")

# Start continuous audio stream
def start_listening():
    # Define a larger blocksize to reduce the frequency of callback invocations
    stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=22050, device=device_id)  # Adjust blocksize as needed
    with stream:
        print("Listening...")
        while True:
            sd.sleep(1000)  # Sleep for 1 second in each iteration to keep the stream open

if __name__ == '__main__':
    device_id = 2  # Set to your identified device ID
    start_listening()
