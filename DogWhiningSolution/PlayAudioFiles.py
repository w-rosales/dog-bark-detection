import os
from pydub import AudioSegment
from pydub.playback import play


# Function to play audio file using pydub
def play_audio_file(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        print(f'Playing {audio_path}...')
        play(audio)
    except Exception as e:
        print(f'Failed to play {audio_path}: {e}')


# Function to display menu and get user selection
def display_menu(files):
    print("Select an audio file to play:")
    for i, filename in enumerate(files):
        print(f"{i + 1}. {filename}")
    print(f"{len(files) + 1}. Exit")

    while True:
        try:
            choice = int(input("Enter your choice (number): "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            elif choice == len(files) + 1:
                return None
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# Process all files in the test_data folder
def main():
    test_data_folder = 'C:/Users/William/PycharmProjects/DogWhiningSolution/dog_sound_recognition/dataset/test_data'
    files = [f for f in os.listdir(test_data_folder) if f.endswith('.wav')]

    if not files:
        print("No audio files found in the test_data folder.")
        return

    while True:
        selected_file = display_menu(files)
        if selected_file is None:
            print("Exiting...")
            break

        file_path = os.path.join(test_data_folder, selected_file)
        play_audio_file(file_path)


if __name__ == "__main__":
    main()
