
import pyaudio
import wave
import time

# Impostazioni audio
FORMAT = pyaudio.paInt16      # 16-bit
CHANNELS = 1                  # mono (la maggior parte dei microfoni USB)
RATE = 44100                  # 44.1 kHz
CHUNK = 8192                  # dimensione buffer
RECORD_SECONDS = 10 
OUTPUT_FILE = "registrazione.wav"

audio = pyaudio.PyAudio()

# Trova automaticamente il microfono USB
mic_index = None
for i in range(audio.get_device_count()):
    info = audio.get_device_info_by_index(i)
    if "USB" in info["name"]:
        mic_index = i
        print(f"Microfono USB trovato: {info['name']} (index {i})")

if mic_index is None:
    raise RuntimeError("❌ Nessun microfono USB trovato!")

# Avvia lo stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=mic_index,
                    frames_per_buffer=CHUNK)

print("🎤 Registrazione iniziata per 10 secondi...")

frames = []
start = time.time()

while time.time() - start < RECORD_SECONDS:
    data = stream.read(CHUNK)
    frames.append(data)

print("✅ Registrazione terminata.")

# Ferma e chiudi
stream.stop_stream()
stream.close()
audio.terminate()

# Salva il file WAV
wf = wave.open(OUTPUT_FILE, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"💾 File salvato come: {OUTPUT_FILE}")
