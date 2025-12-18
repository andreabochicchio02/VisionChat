import subprocess

# Increase microphone gain
subprocess.run(["amixer", "-c", "1", "set", "Mic", "100%"], check=False)

import pyaudio
import numpy as np
import threading
import queue
import time
from vosk import Model, KaldiRecognizer
import json

# Audio configuration
RATE = 44100    # Vosk needs this
CHUNK = 4096   #  4096 samples -> 93 ms at 44.1 kHz

class StreamingSpeechRecognizer:
    def __init__(self, model_path="/home/studenti/vosk-model-small-en-us-0.15"):
        print("🧠 Loading Vosk model...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, RATE)
        self.recognizer.SetWords(True)
        
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.running = False
        self.speech_detected = False
        
        # Find USB mic
        self.mic_index = self._find_usb_mic()        
        
    def _find_usb_mic(self):
        """Find USB microphone"""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if "USB" in info["name"]:
                print(f"🎤 Microphone found: {info['name']}")
                return i
        
        raise RuntimeError("❌ No USB microphone found!")
    

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for each audio chunk"""
        # Resample if needed
        self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)   # the input data and a flag telling PyAudio to keep use this callback
    
    def _process_audio(self):
        """Thread that processes audio"""
        while self.running:
            try:
                data = self.audio_queue.get(timeout=0.1)                
                # Check if we're getting audio
                volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                
                # Send to Vosk
                if self.recognizer.AcceptWaveform(data):
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
                    
            except queue.Empty:
                continue
    
    def send_to_llm(self, text):
        """Send text to PC with LLM"""
        # TODO: Implement LLM connection
        pass
    
    def start(self):
        """Start streaming"""
        print("🎙️  Starting audio stream...")
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,  # Record at 44.1kHz
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback
        )
        
        self.running = True
        self.stream.start_stream()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_audio)
        self.process_thread.start()
        
        print("✅ Streaming active! Speak into the microphone...")
        print("   (Press Ctrl+C to stop)\n")
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 User interrupt")
            self.stop()
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        
        self.stream.stop_stream()
        self.stream.close()        
        self.audio.terminate()
        
        self.process_thread.join()
        
        print("✅ Streaming stopped")


if __name__ == "__main__":
    recognizer = StreamingSpeechRecognizer()
    recognizer.start()