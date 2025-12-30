import pyaudio
import whisper
import numpy as np
from scipy import signal

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 4096
RECORD_SECONDS = 10
OUTPUT_TEXT = "transcription.txt"

print("Loading Whisper model...")
model = whisper.load_model("tiny")

audio = pyaudio.PyAudio()

# Automatically find USB microphone
mic_index = None
mic_info = None
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "USB" in info["name"]:
        mic_index = i
        mic_info = info
        print(f"USB microphone found: {info['name']} (index {i})")
        break

if mic_index is None:
    raise RuntimeError("No USB microphone found!")

# Get supported sample rate from device (usually 44100 or 48000)
RATE = int(mic_info["defaultSampleRate"])
print(f"Using sample rate: {RATE} Hz")

# Start stream
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    input_device_index=mic_index,
    frames_per_buffer=CHUNK
)

print("Recording started for 10 seconds...")
frames = []

for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("Recording finished.")

# Stop and close
stream.stop_stream()
stream.close()
audio.terminate()

# -----------------------------
# DIRECT TRANSCRIPTION WITH WHISPER
# -----------------------------
print("Transcribing with Whisper...")

# Convert frames to numpy array
audio_data = b"".join(frames)
audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

# Resample to 16000 Hz if needed (Whisper's optimal rate)
if RATE != 16000:
    print(f"Resampling from {RATE} Hz to 16000 Hz...")
    number_of_samples = int(len(audio_np) * 16000 / RATE)
    audio_np = signal.resample(audio_np, number_of_samples)

# Transcribe directly from numpy array
result = model.transcribe(audio_np, language="it", fp16=False)
text = result["text"]

print("\n=== RECOGNIZED TEXT ===\n")
print(text)