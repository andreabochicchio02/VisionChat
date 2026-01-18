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

# Audio Configuration
RATE = 44100
CHUNK = 4096
MIC_CARD = "1"
MIC_CONTROL_NAME = "Mic"
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"

def configure_microphone_gain() -> None:
    """
    Try to set microphone gain using `amixer`. Fail silently if not available.
    This is best-effort and non-blocking (does not raise on failure).
    """
    try:
        subprocess.run(["amixer", "-c", MIC_CARD, "set", MIC_CONTROL_NAME, "85%"], check=False)
    except Exception:
        pass

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

    def __init__(self, detection_queue: mp.Queue, ui_notification_queue: queue.Queue):
        self.detection_queue = detection_queue
        self.ui_notification_queue = ui_notification_queue
        
        # Initialize Vosk speech recognition
        print("Loading speech recognition model")
        self.model = Model(VOSK_MODEL_PATH)                     # Load Vosk speech recognition model
        self.recognizer = KaldiRecognizer(self.model, RATE)     # Create recognizer with sample rate
        self.recognizer.SetWords(True)                          # Enable word-level recognition

        # Audio streaming setup
        self.audio = pyaudio.PyAudio()                          # Library that allows you to interface with audio devices 
        self.audio_queue = queue.Queue()                        # Queue for audio data
        self.stream = None                                      # Placeholder for audio stream
        
        # State management
        self.running = False
        self.listening = False

        # Initialize LLM client
        self.llm = LLMClient()

        # Find and configure microphone
        self.mic_index = self.find_usb_microphone()

        # Thread for listening to proactive alerts from detection process
        self.alert_thread = None
        self.alert_enabled = []    # List of class names to report
    
    def find_usb_microphone(self) -> int:
        """Locate USB microphone device"""

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
    
    def record_and_recognize(self) -> str:
        """
        Background thread that processes audio data
        Handles both partial and final speech recognition results
        """

        self.recognizer.Reset()
        recognized_text = ""
        self.listening = True

        print("Listening...")
        start_time = time.time()
        
        while self.listening:
            # Timeout after 10 seconds if no speech detected
            if time.time() - start_time > 10 and recognized_text == "":
                 self.listening = False
                 return ""

            try:
                audio_data = self.audio_queue.get(timeout=0.5)

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
        """Speak a response and pause audio processing during speech"""
        text_to_speech(text)
        print("Speech complete")

    def alert_listener(self) -> None:
        """Background thread that listens for alerts from the object detection process"""
        while self.running:
            try:
                detection = self.detection_queue.get(timeout=0.5)

                visible_classes = {obj.class_name for obj in detection}
            
                for cls in list(visible_classes):

                    if cls in self.alert_enabled:
                        # Get the first detected object of this class
                        obj = next(o for o in detection if o.class_name == cls)
                        timestamp = time.time()

                        # Prepare notification message
                        class_name = obj.class_name
                        pos = obj.get_position_description()
                        conf = obj.confidence
                        msg = f"Notification: {class_name} detected ({conf:.0f}% confidence) at {pos}."

                        print(f"[NOTIFICATION] {msg} (timestamp: {time.strftime('%H:%M:%S', time.localtime(timestamp))})\n", flush=True)

                        # Send notification to the UI queue
                        try:
                            self.ui_notification_queue.put({"notification": msg})
                        except Exception as e:
                            print(f"Error sending notification to UI: {e}")

                        # Speak the notification
                        self.speak_response(msg)

                        # Disable alert for this class to prevent repeated notifications
                        self.alert_enabled.remove(cls)

            except queue.Empty:
                # No detection available, continue the loop
                continue

    def start_background_services(self):
        """Start threads and audio stream but NOT the main interaction loop"""
        self.running = True
        self.start_alert_listener_thread()
        self.open_audio_stream()
        configure_microphone_gain()

    def start_alert_listener_thread(self) -> None:
        """Start background thread to listen for proactive alerts."""
        self.alert_thread = threading.Thread(target=self.alert_listener, daemon=True)
        self.alert_thread.start()

    def open_audio_stream(self) -> None:
        """Open and start the audio stream using current microphone index."""

        print("Starting audio stream...")
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

    def handle_alert_request(self, recognized_text: str) -> str:
        """
        Check whether the recognized text is an alert command and handle it.
        """
        alert_info = self.llm.detect_alert_request(recognized_text)
        
        if isinstance(alert_info, dict) and alert_info.get("is_alert_request"):
            valid_objects = [obj for obj in alert_info.get("target_objects", []) if obj in CLASSES]
            
            if valid_objects:
                for obj in valid_objects:
                    if obj not in self.alert_enabled:
                        self.alert_enabled.append(obj)
            
                return f"I will notify you when I see: {', '.join(valid_objects)}."

            return "I understood you want an alert, but I don't recognize those objects."
        return ""

    def process_single_interaction_streaming(self):
        """
        Generator that yields updates for streaming response to UI.
        
        Flow:
        - Listen to user input
        - Yield user text immediately after recognition
        - Process with LLM
        - Yield assistant response
        - Speak response on Raspberry Pi
        """
        print("Starting interaction cycle via UI trigger")
        
        # 1. Listen for user input
        user_text = self.record_and_recognize()
        if not user_text:
            yield {"error": "No speech detected."}
            return

        print(f"User said: {user_text}")

        # Send user text to UI immediately after recognition
        yield {"user": user_text}

        # 2. Check for alert requests
        alert_resp = self.handle_alert_request(user_text)
        if alert_resp:
            # Send assistant response for alert
            yield {"assistant": alert_resp}

            # Speak the response
            self.speak_response(alert_resp)
            return

        # 3. Get latest detected objects
        try:
            detected_objects = self.detection_queue.get(timeout=2)
        except queue.Empty:
            detected_objects = []

        # 4. Generate response from LLM
        response = self.llm.generate_response(detected_objects, user_text)
        
        # Send assistant response to UI
        yield {"assistant": response}

        # 5. Speak the response on Raspberry Pi
        self.speak_response(response)

    def stop(self):
        """Stop all services and clean up resources"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()