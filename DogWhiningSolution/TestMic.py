import sounddevice as sd
import scipy.io.wavfile as wav
import os

# Select the correct microphone device ID
device_id = 2  # Set to your identified device ID

# Function to record audio from the microphone and save it as a WAV file
def record_audio(duration, filename, device_id):
    fs = 44100  # Sample rate
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device_id)
    sd.wait()  # Wait until recording is finished
    wav.write(filename, fs, audio)
    print(f"Recorded audio saved as {filename}")

# Function to play back the recorded audio
def play_audio(filename):
    fs, data = wav.read(filename)
    print(f"Playing back audio from {filename}")
    sd.play(data, fs)
    sd.wait()  # Wait until playback is finished

# Paths
audio_filename = 'test_audio.wav'

# Record and play back audio
record_audio(5, audio_filename, device_id)
if os.path.isfile(audio_filename):
    play_audio(audio_filename)
else:
    print(f'File not found: {audio_filename}')
