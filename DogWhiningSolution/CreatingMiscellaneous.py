import os
import random
import time
from pydub import AudioSegment

# Function to randomly select and combine miscellaneous samples
def combine_miscellaneous_samples(misc_folder, output_folder, num_samples=3, output_file='synthesized_misc'):
    # Get list of all miscellaneous files in the misc folder
    misc_files = [f for f in os.listdir(misc_folder) if f.endswith('.wav')]

    # Ensure there are enough samples to choose from
    if len(misc_files) < num_samples:
        print(f"Not enough miscellaneous files to combine. Found {len(misc_files)}, but need at least {num_samples}.")
        return None

    # Randomly select miscellaneous samples
    selected_miscs = random.sample(misc_files, num_samples)

    # Combine the selected miscellaneous samples
    combined_misc = AudioSegment.silent(duration=1000)  # Start with a second of silence
    for misc_file in selected_miscs:
        misc_path = os.path.join(misc_folder, misc_file)
        misc = AudioSegment.from_file(misc_path)
        combined_misc = combined_misc.overlay(misc)

    # Normalize the combined miscellaneous
    combined_misc = combined_misc.normalize()

    return combined_misc

# Function to save audio segments with a unique name
def save_combined_misc(combined_misc, output_folder, output_file_base):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate a unique filename using timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_file = f'{output_file_base}_{timestamp}.wav'
    output_path = os.path.join(output_folder, output_file)

    # Save the combined miscellaneous as a new file
    combined_misc.export(output_path, format='wav')
    print(f'Synthesized miscellaneous saved as {output_path}')

# Paths
misc_folder = 'dog_sound_recognition/dataset/miscellaneous'
output_folder = 'dog_sound_recognition/dataset/test_data'

# Generate 500 unique synthesized miscellaneous files
for i in range(400):
    combined_misc = combine_miscellaneous_samples(misc_folder, output_folder)
    if combined_misc:
        save_combined_misc(combined_misc, output_folder, f'synthesized_misc_{i + 1}')
