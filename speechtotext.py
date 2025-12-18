import pyaudio
import wave
import time
import whisper
import numpy as np

# Impostazioni audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # Whisper funziona meglio a 16kHz
CHUNK = 4096
RECORD_SECONDS = 10

OUTPUT_WAV = "registrazione.wav"
OUTPUT_TEXT = "trascrizione.txt"

print("🧠 Caricamento modello Whisper...")
model = whisper.load_model("tiny")  # "tiny" = veloce su Raspberry

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
    data = stream.read(CHUNK, exception_on_overflow=False)
    frames.append(data)

print("✅ Registrazione terminata.")

# Ferma e chiudi
stream.stop_stream()
stream.close()
audio.terminate()

# -----------------------------
# SALVA IL FILE WAV
# -----------------------------
wf = wave.open(OUTPUT_WAV, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()

print(f"💾 File audio salvato come: {OUTPUT_WAV}")

# -----------------------------
# TRASCRIZIONE CON WHISPER
# -----------------------------
print("🧠 Trascrizione con Whisper in corso...")

# Usa il file WAV direttamente (più affidabile)
result = model.transcribe(OUTPUT_WAV, language="it", fp16=False)

text = result["text"]

print("\n=== TESTO RICONOSCIUTO ===\n")
print(text)

# Salva su file
with open(OUTPUT_TEXT, "w", encoding="utf-8") as f:
    f.write(text)

print(f"\n💾 Testo salvato in: {OUTPUT_TEXT}")