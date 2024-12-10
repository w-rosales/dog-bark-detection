import os
import random
import time
from pydub import AudioSegment


# Function to randomly select and combine whine samples
def combine_whine_samples(whine_folder, output_folder, num_samples=3, output_file='synthesized_whine'):
    # Get list of all whine files in the whine folder
    whine_files = [f for f in os.listdir(whine_folder) if f.endswith('.wav')]

    # Ensure there are enough samples to choose from
    if len(whine_files) < num_samples:
        print(f"Not enough whine files to combine. Found {len(whine_files)}, but need at least {num_samples}.")
        return None

    # Randomly select whine samples
    selected_whines = random.sample(whine_files, num_samples)

    # Combine the selected whine samples
    combined_whine = AudioSegment.silent(duration=1000)  # Start with a second of silence
    for whine_file in selected_whines:
        whine_path = os.path.join(whine_folder, whine_file)
        whine = AudioSegment.from_file(whine_path)
        combined_whine = combined_whine.overlay(whine)

    # Normalize the combined whine
    combined_whine = combined_whine.normalize()

    return combined_whine


# Function to save audio segments with a unique name
def save_combined_whine(combined_whine, output_folder, output_file_base):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique filename using timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_file = f'{output_file_base}_{timestamp}.wav'
    output_path = os.path.join(output_folder, output_file)

    # Save the combined whine as a new file
    combined_whine.export(output_path, format='wav')
    print(f'Synthesized whine saved as {output_path}')


# Paths
whine_folder = 'dog_sound_recognition/dataset/whine'
output_folder = 'dog_sound_recognition/dataset/test_data'

# Generate 10 unique synthesized whine files
for i in range(500):
    combined_whine = combine_whine_samples(whine_folder, output_folder)
    if combined_whine:
        save_combined_whine(combined_whine, output_folder, f'synthesized_whine_{i + 1}')
