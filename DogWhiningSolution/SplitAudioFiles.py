import os
from pydub import AudioSegment


# Function to split audio file into segments of 5 seconds
def split_audio_file(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_in_seconds = len(audio) / 1000
        segment_duration = 1 * 1000  # 5 seconds in milliseconds

        for i in range(0, len(audio), segment_duration):
            segment = audio[i:i + segment_duration]
            segment_file_name = f'{os.path.splitext(os.path.basename(audio_path))[0]}_part{i // segment_duration + 1}.wav'
            segment.export(os.path.join(os.path.dirname(audio_path), segment_file_name), format='wav')
            print(f'Segment {i // segment_duration + 1} saved as {segment_file_name}')

    except Exception as e:
        print(f'Failed to split {audio_path}: {e}')


# Process all files in the test_data folder
def main():
    test_data_folder = 'C:/Users/William/PycharmProjects/DogWhiningSolution/dog_sound_recognition/dataset/test_data'
    files = [f for f in os.listdir(test_data_folder) if f.endswith('.wav')]

    if not files:
        print("No audio files found in the test_data folder.")
        return

    for filename in files:
        file_path = os.path.join(test_data_folder, filename)
        split_audio_file(file_path)


if __name__ == "__main__":
    main()
