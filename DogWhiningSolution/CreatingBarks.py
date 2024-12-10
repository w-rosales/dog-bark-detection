import os
import random
import time
from pydub import AudioSegment


# Function to randomly select and combine bark samples
def combine_bark_samples(bark_folder, output_folder, num_samples=3, output_file='synthesized_bark'):
    # Get list of all bark files in the bark folder
    bark_files = [f for f in os.listdir(bark_folder) if f.endswith('.wav')]

    # Ensure there are enough samples to choose from
    if len(bark_files) < num_samples:
        print(f"Not enough bark files to combine. Found {len(bark_files)}, but need at least {num_samples}.")
        return None

    # Randomly select bark samples
    selected_barks = random.sample(bark_files, num_samples)

    # Combine the selected bark samples
    combined_bark = AudioSegment.silent(duration=1000)  # Start with a second of silence
    for bark_file in selected_barks:
        bark_path = os.path.join(bark_folder, bark_file)
        bark = AudioSegment.from_file(bark_path)
        combined_bark = combined_bark.overlay(bark)

    # Normalize the combined bark
    combined_bark = combined_bark.normalize()

    return combined_bark


# Function to save audio segments with a unique name
def save_combined_bark(combined_bark, output_folder, output_file_base):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique filename using timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_file = f'{output_file_base}_{timestamp}.wav'
    output_path = os.path.join(output_folder, output_file)

    # Save the combined bark as a new file
    combined_bark.export(output_path, format='wav')
    print(f'Synthesized bark saved as {output_path}')


# Paths
bark_folder = 'dog_sound_recognition/dataset/bark'
output_folder = 'dog_sound_recognition/dataset/test_data'

# Generate 10 unique synthesized bark files
for i in range(500):
    combined_bark = combine_bark_samples(bark_folder, output_folder)
    if combined_bark:
        save_combined_bark(combined_bark, output_folder, f'synthesized_bark_{i + 1}')

