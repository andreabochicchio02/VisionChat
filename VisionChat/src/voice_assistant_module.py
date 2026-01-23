import json
import subprocess
import threading
import time
import queue
import multiprocessing as mp
import pyaudio
import os
from vosk import Model, KaldiRecognizer

from detected_object_module import CLASSES
from chatLLM import LLMClient

# Audio Configuration
RATE = 44100
CHUNK = 4096
MIC_CARD = "1"
MIC_CONTROL_NAME = "Mic"
VOSK_MODEL_PATH = {
    "en": "models/vosk-model-small-en-us-0.15",
    "it": "models/vosk-model-small-it-0.22",
}

# Load messages from JSON file
def load_messages():
    """Load language-specific messages from `text.json`."""
    prompts_path = os.path.join(os.path.dirname(__file__), 'text.json')
    with open(prompts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {lang: data['text'][lang]['messages'] for lang in data['text']}

MESSAGES = load_messages()

def configure_microphone_gain() -> None:
    """
    Try to set microphone gain using `amixer`. Fail silently if not available.
    """
    try:
        subprocess.run(["amixer", "-c", MIC_CARD, "set", MIC_CONTROL_NAME, "85%"], check=False)
    except Exception:
        pass


def text_to_speech_IT(text: str) -> None:
    try:
        subprocess.run(["pico2wave", "-l", "it-IT", "-w", "/tmp/tts.wav", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )
        subprocess.run(["aplay", "/tmp/tts.wav"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("Failed to run pico TTS")

def text_to_speech_EN(text: str) -> None:
    try:
        subprocess.run(["pico2wave", "-l", "en-US", "-w", "/tmp/tts.wav", text], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL )
        subprocess.run(["aplay", "/tmp/tts.wav"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        print("Failed to run pico TTS")


class VoiceAssistant:
    """Handles speech recognition, alerts and interaction with object detection"""

    def __init__(self, detection_queue: mp.Queue, ui_notification_queue: queue.Queue, lang: str):
        self.detection_queue = detection_queue
        self.ui_notification_queue = ui_notification_queue
        self.language = lang
        
        # Initialize Vosk speech recognition
        print("Loading speech recognition model")
        self.model = Model(VOSK_MODEL_PATH[self.language])      # Load Vosk speech recognition model
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
        self.llm = LLMClient(self.language, max_history=0)

        # Find and configure microphone
        self.mic_index = self.find_usb_microphone()

        # Thread for listening to proactive alerts from detection process
        self.alert_thread = None
        self.alert_enabled = []    # List of class names to report
        self.motion_alert_enabled = False  # Flag for motion detection alerts

        # Stores the most recent object detections received from the vision process.
        # Required because detections are updated by the listener thread
        # and read by other threads (e.g. LLM interaction), preventing race conditions.
        self.last_detections = []
        self.last_detections_lock = threading.Lock()
    
    def find_usb_microphone(self) -> int:
        """Locate USB microphone device"""

        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if "USB" in device_info.get("name", ""):
                print(f"Found microphone: {device_info['name']}")
                return i

        raise RuntimeError("No USB microphone detected")
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for audio stream
        Queues audio data only when listening is active
        """
        # Only queue audio when actively listening
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
                        print("\n")
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
        if self.language == "it":
            text_to_speech_IT(text)
        else:
            text_to_speech_EN(text)
        print("Speech complete")

    def listener(self) -> None:
        """Background thread that listens for alerts from the object detection process"""
        while self.running:
            try:
                detection_data = self.detection_queue.get(timeout=0.5)
                
                # Extract objects and motion status from detection data
                detected_objects = detection_data.get('objects', [])
                motion_detected = detection_data.get('motion_detected', False)

                # Required lock because detections are shared by the listener thread
                # and other thread (e.g. LLM interaction)
                with self.last_detections_lock:
                    self.last_detections = detected_objects

                # Check for motion alerts
                if self.motion_alert_enabled and motion_detected:
                    msg = MESSAGES[self.language]['motion_notification']
                    print(f"[MOTION NOTIFICATION] {msg}\n", flush=True)
                    
                    try:
                        self.ui_notification_queue.put({"notification": msg})
                    except Exception as e:
                        print(f"Error sending motion notification to UI: {e}")
                    
                    self.speak_response(msg)
                    self.motion_alert_enabled = False  # One-shot notification

                # Check for object alerts
                visible_classes = {obj.class_name for obj in detected_objects}
            
                for cls in list(visible_classes):
                    if cls in self.alert_enabled:
                        # Get the first detected object of this class
                        obj = next(o for o in detected_objects if o.class_name == cls)

                        msg = MESSAGES[self.language]['notification'].format(
                            class_name=obj.class_name,
                            conf=obj.confidence,
                            pos=obj.get_position_description(self.language)
                        )

                        print(f"[NOTIFICATION] {msg}\n", flush=True)

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
        self.alert_thread = threading.Thread(target=self.listener, daemon=True)
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
        """Check whether the recognized text is an alert command and handle it."""
        # Ask the LLM whether the user requested alerts and which objects

        alert_info = self.llm.detect_alert_request(recognized_text)

        if isinstance(alert_info, dict):
            # Check for motion alert request (can be independent of is_alert_request)
            if alert_info.get("is_motion_request"):
                self.motion_alert_enabled = True
                return MESSAGES[self.language]['motion_alert_response']
            
            # Handle object-based alerts (requires is_alert_request)
            if alert_info.get("is_alert_request"):
                target_objects = alert_info.get("target_objects", [])
                valid_objects = [obj for obj in target_objects if obj in CLASSES[self.language]]
                
                if valid_objects:
                    for obj in valid_objects:
                        if obj not in self.alert_enabled:
                            self.alert_enabled.append(obj)
                
                    return MESSAGES[self.language]['alert_response'].format(
                        objects=', '.join(valid_objects)
                    )

                return MESSAGES[self.language]['alert_no_objects']
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

        # Send user text to UI immediately after recognition
        yield {"user": user_text}

        # 2. Check for alert requests (enable/disable notifications)
        alert_resp = self.handle_alert_request(user_text)
        if alert_resp:
            # Send assistant response for alert
            yield {"assistant": alert_resp}

            # Speak the response
            self.speak_response(alert_resp)
            return

        # 3. Get latest detected objects snapshot
        with self.last_detections_lock:
            detected_objects = list(self.last_detections)        
        
        full_response_text = ''

        # 4. Stream tokens from LLM to UI
        for token in self.llm.generate_response(detected_objects, user_text):
            full_response_text += token
            yield {"token": token}

        # 5. Speak the response on Raspberry Pi
        self.speak_response(full_response_text)

    def stop(self):
        """Stop all services and clean up resources"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        # Shutdown of alert thread
        if self.alert_thread and self.alert_thread.is_alive():
            try:
                self.alert_thread.join(timeout=0.5)
            except Exception:
                pass

        self.audio.terminate()