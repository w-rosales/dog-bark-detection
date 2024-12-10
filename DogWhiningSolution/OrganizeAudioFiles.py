import os


# Function to rename files in a folder sequentially
def rename_files_in_folder(folder_path, prefix):
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    files.sort()  # Sort files to maintain order
    temp_folder = os.path.join(folder_path, 'temp')
    os.makedirs(temp_folder, exist_ok=True)

    # Temporarily move all files to a temp folder to avoid conflicts
    for filename in files:
        old_file_path = os.path.join(folder_path, filename)
        temp_file_path = os.path.join(temp_folder, filename)
        os.rename(old_file_path, temp_file_path)

    # Rename files back to the original folder with new names
    for i, filename in enumerate(sorted(os.listdir(temp_folder))):
        new_filename = f'{prefix}_{i + 1}.wav'
        temp_file_path = os.path.join(temp_folder, filename)
        new_file_path = os.path.join(folder_path, new_filename)
        os.rename(temp_file_path, new_file_path)
        print(f'Renamed {filename} to {new_filename}')

    os.rmdir(temp_folder)  # Remove the temporary folder


# Function to organize all folders
def organize_audio_files(base_path):
    folders = ['bark', 'whine', 'miscellaneous']

    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        if os.path.exists(folder_path):
            rename_files_in_folder(folder_path, folder)
        else:
            print(f'Folder {folder_path} does not exist.')


# Main function
def main():
    base_path = 'dog_sound_recognition/dataset'  # Adjust base path as needed
    organize_audio_files(base_path)


if __name__ == "__main__":
    main()
