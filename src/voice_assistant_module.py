import json
import subprocess
import threading
import time
import queue
import multiprocessing as mp
from typing import List
import pyaudio
import requests
import sys
from vosk import Model, KaldiRecognizer

from detected_object_module import DetectedObject, CLASSES
from chatLLM import LLMClient


# Audio stream configuration 
RATE = 44100
CHUNK = 4096
MIC_CARD = "1"
MIC_CONTROL_NAME = "Mic"

# Speech recognition and synthesis configuration
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
ESPEAK_VOICE = "en-us"
ESPEAK_SPEED = "140"

# Language Model configuration
URL = "http://10.75.235.144:11434/api/generate"
MODEL = "llama3.2:3b"


# Utility functions

def configure_microphone_gain() -> None:
    """
    Try to set microphone gain using `amixer`. Fail silently if not available.
    This is best-effort and non-blocking (does not raise on failure).
    """
    try:
        subprocess.run(["amixer", "-c", MIC_CARD, "set", MIC_CONTROL_NAME, "100%"], check=False)
    except Exception:
        print("Failed to run amixer; continuing without changing mic gain")


def text_to_speech(text: str) -> None:
    """
    Synthesize speech with espeak (blocking until finished).
    Keeps stdout/stderr quiet by redirecting to DEVNULL.
    """
    try:
        subprocess.run(["espeak", "-v", "en-us", "-s", "140", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("Failed to run espeak")



class VoiceAssistant:
    """Handles speech recognition, alerts and interaction with object detection"""

    def __init__(self, detection_queue: mp.Queue, command_queue: mp.Queue, alert_queue: mp.Queue):
        self.detection_queue = detection_queue
        self.command_queue = command_queue
        self.alert_queue = alert_queue
        
        # Initialize Vosk speech recognition
        print("Loading speech recognition model")
        self.model = Model(VOSK_MODEL_PATH)                     # load Vosk speech recognition model
        self.recognizer = KaldiRecognizer(self.model, RATE)     # create recognizer with sample rate
        self.recognizer.SetWords(True)                          # enable word-level recognition

        # Audio streaming setup
        self.audio = pyaudio.PyAudio()                          # library that allows you to interface with audio devices 
        self.audio_queue = queue.Queue()                        # queue for audio data
        self.stream = None                                      # placeholder for audio stream
        
        # State management
        self.running = False
        self.listening = False
        
        # Find and configure microphone
        self.mic_index = self.find_usb_microphone()

        # thread for listening to proactive alerts from detection process
        self.alert_thread = None
    
    def find_usb_microphone(self) -> int:
        """
        Locate USB microphone device
        """
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if "USB" in device_info["name"]:
                print(f"Found microphone: {device_info['name']}")
                return i
        
        raise RuntimeError("No USB microphone detected")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for audio stream
        Queues audio data only when listening is active
        """

        if self.listening:  
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def record_and_recognize(self) -> None:
        """
        Background thread that processes audio data
        Handles both partial and final speech recognition results
        """
                
        self.recognizer.Reset()
        recognized_text = ""

        self.listening = True
        
        while self.listening:
            
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Check if we have a complete phrase
                if self.recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.recognizer.Result())
                    recognized_text = result.get("text", "")

                    if recognized_text:
                        break

                else:
                    # Handle partial recognition results
                    partial = json.loads(self.recognizer.PartialResult())
                    text = partial.get("partial", "")
                    
                    if text:
                        print(f"\r🗣️  {text}", end="", flush=True)
            
            except queue.Empty:
                continue

        self.listening = False
        return recognized_text
    
    
    def speak_response(self, text: str) -> None:
        """
        Speak a response and pause audio processing during speech
        
        Args:
            text: Text to convert to speech
        """
        
        # Perform text-to-speech
        text_to_speech(text)
        
        print("Speech complete")


    def alert_listener(self) -> None:
        """Background thread that listens for alerts from the object detection process"""

        while self.running:
            try:
                alert = self.alert_queue.get(timeout=0.5)

                if not alert:
                    continue
                else:
                    obj: DetectedObject = alert['object']
                    timestamp: float = alert.get('timestamp', time.time())

                    class_name = obj.class_name
                    pos = obj.get_position_description()
                    conf = obj.confidence

                    msg = f"Notification: {class_name} detected ({conf:.0f}% confidence) at {pos}."
                    print(f"\n[NOTIFICATION] {msg} (timestamp: {time.strftime('%H:%M:%S', time.localtime(timestamp))})", flush=True)

                    # Speak the notification
                    self.speak_response(msg)

            except queue.Empty:
                continue


    def prepare_llm(self) -> LLMClient:
        """Create and return an LLM client instance."""
        return LLMClient()

    def start_alert_listener_thread(self) -> None:
        """Start background thread to listen for proactive alerts."""
        self.alert_thread = threading.Thread(target=self.alert_listener, daemon=True)
        self.alert_thread.start()

    def open_audio_stream(self) -> None:
        """Open and start the audio stream using current microphone index."""

        print("Starting audio stream...")

        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=CHUNK,
            stream_callback=self.audio_callback
        )

        self.stream.start_stream()

        print("Voice Assistant active!")

    def handle_alert_request(self, recognized_text: str, llm: LLMClient) -> bool:
        """
        Check whether the recognized text is an alert command and handle it.
        """
        alert_info = llm.detect_alert_request(recognized_text)

        if isinstance(alert_info, dict) and alert_info.get("is_alert_request"):
            raw_objects = alert_info.get("target_objects", [])
            valid_objects = []
            invalid_objects = []

            for obj in raw_objects:
                clean_obj = obj.strip().lower()

                # Keep existing behavior: attempt to check CLASSES if present
                if clean_obj in CLASSES:
                    valid_objects.append(clean_obj)
                else:
                    invalid_objects.append(clean_obj)

            response_text = ""
            if valid_objects:
                objs_str = ", ".join(valid_objects)
                response_text += f"Understood. I will notify you when I detect: {objs_str}. "

                # Send command to object detection process to set alerts
                self.command_queue.put(("SET_ALERTS", valid_objects))

            if invalid_objects:
                objs_str = ", ".join(invalid_objects)
                response_text += f"I am unable to detect the following object: {objs_str}."
            
            if not response_text:
                response_text = "I understood you want an alert, but I couldn't identify which object specifically."

            print(response_text)
            self.speak_response(response_text)
            return True

        return False

    def process_user_interaction(self, llm: LLMClient) -> None:
        """Main loop that prompts the user, records speech, and processes requests."""

        try:
            while self.running:
                input("Press ENTER to speak... ")

                recognized_text = self.record_and_recognize()

                # If text was recognized, process the request
                if recognized_text:
                    print(f"\nRecognized Input: {recognized_text}")

                    # Verify if the user just wants a notification upon object detection
                    if self.handle_alert_request(recognized_text, llm):
                        continue

                    # Request latest object detections
                    self.command_queue.put("GET_DETECTIONS")

                    # Wait for detection results
                    try:
                        detected_objects: List[DetectedObject] = self.detection_queue.get(timeout=1)
                    except queue.Empty:
                        detected_objects = []
                        print("Detection data not available")

                    # Generate response
                    response = llm.generate_response(detected_objects, recognized_text)

                    self.speak_response(response)

                else:
                    print("No voice input recognized. Try again")

                print("="*50, "\n")

        except KeyboardInterrupt:
            print("User interrupted")
            self.stop()

    def start(self) -> None:
        """Start the voice assistant by initializing components and entering main loop."""

        self.running = True

        # Initialize LLM client
        llm = self.prepare_llm()

        # start background thread to listen for proactive alerts
        self.start_alert_listener_thread()

        # Open audio stream
        self.open_audio_stream()

        # Enter main interaction loop
        self.process_user_interaction(llm)

        if self.alert_thread and self.alert_thread.is_alive():
            self.alert_thread.join()
        
        print("Voice assistant stopped")


