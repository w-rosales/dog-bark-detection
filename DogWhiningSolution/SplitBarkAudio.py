import os
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent


# Function to split audio file into segments based on amplitude threshold
def split_audio_file(audio_path, min_db=-20, min_len=500):
    try:
        audio = AudioSegment.from_file(audio_path)
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_len, silence_thresh=min_db)

        # Extract segments
        segments = []
        for start, end in nonsilent_ranges:
            segment = audio[start:end]
            segments.append(segment)

        return segments

    except Exception as e:
        print(f'Failed to split {audio_path}: {e}')
        return []


# Function to save audio segments
def save_segments(segments, base_path, base_filename):
    for i, segment in enumerate(segments):
        segment_file_name = f'{base_filename}_bark_{i + 1}.wav'
        segment.export(os.path.join(base_path, segment_file_name), format='wav')
        print(f'Saved segment as {segment_file_name}')


# Process all files in the test_data folder
def main():
    test_data_folder = 'dog_sound_recognition/dataset/test_data'
    output_folder = 'dog_sound_recognition/dataset/test_data'
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(test_data_folder) if f.endswith('.wav')]

    if not files:
        print("No audio files found in the test_data folder.")
        return

    for filename in files:
        file_path = os.path.join(test_data_folder, filename)
        base_filename = os.path.splitext(filename)[0]
        segments = split_audio_file(file_path, min_db=-20, min_len=500)
        save_segments(segments, output_folder, base_filename)


if __name__ == "__main__":
    main()
