import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Function to extract features from audio
def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f'Error processing {audio_path}: {e}')
        return None

# Function to load and preprocess the dataset
def load_dataset(base_path):
    X = []
    y = []
    labels = {'bark': 0, 'whine': 1, 'miscellaneous': 2}  # Ensure these match the subfolder names

    for label in labels:
        folder = os.path.join(base_path, label)
        for filename in os.listdir(folder):
            if filename.endswith('.wav'):
                file_path = os.path.join(folder, filename)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    y.append(labels[label])

    return np.array(X), np.array(y)

# Base path to your dataset
base_path = 'dog_sound_recognition/dataset'
X, y = load_dataset(base_path)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoaders for training and validation
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Increased batch size to 8

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)  # Increased batch size to 8

# Model definition
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

model = DogSoundModel()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Reduced learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)  # Learning rate scheduler

# Early stopping parameters
early_stop_patience = 10
early_stop_counter = 0
best_val_loss = float('inf')

# Model training with validation, early stopping, and learning rate scheduling
for epoch in range(50):  # Number of epochs
    # Training phase
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    # Adjust learning rate
    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'dog_sound_model.pth')
        print(f'Saving model with validation loss: {avg_val_loss}')
        early_stop_counter = 0
    else:
        early_stop_counter += 1

    if early_stop_counter >= early_stop_patience:
        print('Early stopping triggered')
        break

    print(f'Epoch {epoch+1}/50, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

print('Model saved successfully to dog_sound_model.pth.')
