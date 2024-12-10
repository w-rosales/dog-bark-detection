import pyaudio

# Initialize PyAudio
audio = pyaudio.PyAudio()

# List all audio input devices
print("Available audio input devices:")
for i in range(audio.get_device_count()):
    device_info = audio.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']}")

# Close PyAudio
audio.terminate()
