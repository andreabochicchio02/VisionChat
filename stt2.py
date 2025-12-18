import subprocess

# Increase microphone gain
subprocess.run(["amixer", "-c", "1", "set", "Mic", "100%"], check=False)

import pyaudio
import numpy as np
import time
from vosk import Model, KaldiRecognizer
import json

# Audio configuration
HARDWARE_RATE = 44100  # Your mic's native rate
TARGET_RATE = 44100    # Vosk needs this
CHUNK = 4096   #  4096 samples -> 93 ms at 44.1 kHz

class StreamingSpeechRecognizer:
    def __init__(self, model_path="/home/studenti/vosk-model-small-en-us-0.15"):
        print("🧠 Loading Vosk model...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, TARGET_RATE)
        self.recognizer.SetWords(True)
        
        self.audio = pyaudio.PyAudio()
        self.speech_detected = False
        
        # Find USB mic
        self.mic_index = self._find_usb_mic()        
        print(f"🎤 Using sample rate: {HARDWARE_RATE}Hz -> {TARGET_RATE}Hz")
        
    def _find_usb_mic(self):
        """Find USB microphone"""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if "USB" in info["name"]:
                print(f"🎤 Microphone found: {info['name']}")
                return i
        
        raise RuntimeError("❌ No USB microphone found!")
    
    def _resample(self, audio_data):
        """Downsample from 44.1kHz to 16kHz"""
        audio_np = np.frombuffer(audio_data, dtype=np.int16)
        
        # take every 2.75th sample (44100/16000 = 2.75625)
        num_samples = int(len(audio_np) * TARGET_RATE / HARDWARE_RATE)
        indices = np.linspace(0, len(audio_np) - 1, num_samples)
        resampled = np.interp(indices, np.arange(len(audio_np)), audio_np)
        
        return resampled.astype(np.int16).tobytes()
    
    def send_to_llm(self, text):
        """Send text to PC with LLM"""
        # TODO: Implement LLM connection
        pass
    
    def start(self):
        """Start streaming"""
        print("🎙️  Starting audio stream...")
        
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=HARDWARE_RATE,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=CHUNK
        )
        
        print("✅ Streaming active! Speak into the microphone...")
        print("   (Press Ctrl+C to stop)\n")
        
        try:
            while True:
                # Read audio chunk (blocking call)
                data = stream.read(CHUNK, exception_on_overflow=False)
                
                # Resample
                resampled_data = data
                
                # Check volume
                volume = np.abs(np.frombuffer(resampled_data, dtype=np.int16)).mean()
                
                # Send to Vosk
                if self.recognizer.AcceptWaveform(resampled_data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    
                    if text:
                        print(f"\n✅ PHRASE: {text}")
                        self.send_to_llm(text)
                        self.speech_detected = False
                    else:
                        print(f"\n⚠️  Vosk returned empty result (volume was {volume:.0f})")
                else:
                    # Partial result (live subtitles)
                    partial = json.loads(self.recognizer.PartialResult())
                    text = partial.get("partial", "")
                    if text:
                        if not self.speech_detected:
                            self.speech_detected = True
                            print(f"\n🎯 Speech detected! Volume: {volume:.0f}")
                        print(f"\r🗣️  {text}", end="", flush=True)
                
        except KeyboardInterrupt:
            print("\n🛑 User interrupt")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            print("✅ Streaming stopped")


if __name__ == "__main__":
    recognizer = StreamingSpeechRecognizer()
    recognizer.start()