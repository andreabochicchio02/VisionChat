import json
import subprocess
import threading
import time
import queue
import multiprocessing as mp
from typing import List
import pyaudio
from vosk import Model, KaldiRecognizer

from camera_module import DetectedObject


# Audio stream configuration 
RATE = 44100
CHUNK = 4096
MIC_CARD = "1"
MIC_CONTROL_NAME = "Mic"

# Speech recognition and synthesis configuration
VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"
ESPEAK_VOICE = "en-us"
ESPEAK_SPEED = "140"


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
    """Handles speech recognition and interaction with object detection"""
    
    def __init__(self, detection_queue: mp.Queue, command_queue: mp.Queue):
        self.detection_queue = detection_queue
        self.command_queue = command_queue
        
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
        self.is_speaking = False
        self.speech_detected = False
        
        # Find and configure microphone
        self.mic_index = self._find_usb_microphone()
    
    def _find_usb_microphone(self) -> int:
        """
        Locate USB microphone device
        """
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if "USB" in device_info["name"]:
                print(f"Found microphone: {device_info['name']}")
                return i
        
        raise RuntimeError("No USB microphone detected")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Callback function for audio stream
        Queues audio data unless the system is currently speaking
        """

        # Only process microphone input when the assistant is not speaking, 
        # to avoid an echo loop during speech output
        if not self.is_speaking:    
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio_stream(self) -> None:
        """
        Background thread that processes audio data
        Handles both partial and final speech recognition results
        """
        while self.running:
            # Pause processing while speaking
            if self.is_speaking:
                time.sleep(0.1)         #TODO we could increase this value
                continue
            
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # Check if we have a complete phrase
                if self.recognizer.AcceptWaveform(audio_data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    
                    if text:
                        print(f"\n>>> Request: {text}")
                        self._handle_user_input(text)
                        self.speech_detected = False
                else:
                    # Handle partial recognition results
                    partial = json.loads(self.recognizer.PartialResult())
                    text = partial.get("partial", "")
                    
                    if text:
                        if not self.speech_detected:        # speech detected, user starts talking
                            self.speech_detected = True
                        print(f"\r🗣️  {text}", end="", flush=True)
            
            except queue.Empty:
                continue
    
    def _handle_user_input(self, input: str) -> None:
        """
        Process recognized input (text) and generate response
        """
        print("Processing request...")
        
        # Request latest object detections
        self.command_queue.put("GET_DETECTIONS")
        
        # Wait for detection results
        try:
            detected_objects: List[DetectedObject] = self.detection_queue.get(timeout=2.0)
        except queue.Empty:
            detected_objects = []
            print("Detection data not available")
        
        # Log detected objects
        if detected_objects:
            objects_str = ", ".join(str(obj) for obj in detected_objects)
            print(f"Currently visible: {objects_str}")
        else:
            print(f"No objects in view")
        
        # Generate response
        response = self._generate_response(detected_objects, input)
        
        # Speak the response
        print(f">>> Response: {response}")
        self._speak_response(response)
    
    def _generate_response(self, objects: List[DetectedObject], user_text: str) -> str:
        """
        Generate a response based on detected objects
        TODO: Replace with actual LLM integration
        
        Args:
            objects: List of currently detected objects
            user_text: The user's spoken input
            
        Returns:
            Response text to be spoken
        """
        if not objects:
            return "I don't see any objects in the current view"
        
        count = len(objects)
        object_names = ", ".join(obj.class_name for obj in objects)
        
        if count == 1:
            return f"I can see 1 object: {object_names}"
        else:
            return f"I can see {count} objects: {object_names}"
    
    def _speak_response(self, text: str) -> None:
        """
        Speak a response and pause audio processing during speech
        
        Args:
            text: Text to convert to speech
        """
        
        # Pause audio input processing
        self.is_speaking = True
        
        # Clear audio queue to avoid processing stale audio
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        # Perform text-to-speech
        text_to_speech(text)
        
        # Resume audio input processing
        self.is_speaking = False
        print("Speech complete\n")
    
    def start(self) -> None:
        """Start the voice assistant
        
        # EXECUTION
        # 1. Main thread → keeps the program running and handles stop/interrupts.
        # 2. PyAudio callback → PyAudio's internal thread that receives audio data and puts it into the queue.
        # 3. _process_audio_stream thread → separate thread that reads from the queue, performs speech recognition, and generates responses.
        """
        print("Starting audio stream...")
        
        # Open audio stream
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            input_device_index=self.mic_index,
            frames_per_buffer=CHUNK,
            stream_callback=self._audio_callback
        )
        
        self.running = True
        self.stream.start_stream()
        
        # Start audio processing thread
        self.process_thread = threading.Thread(target=self._process_audio_stream, daemon=True)
        self.process_thread.start()
        
        print("Voice assistant active! Speak into the microphone...")
        
        # Main loop
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("User interrupted")
            self.stop()
    
    def stop(self) -> None:
        """Shutdown the voice assistant"""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        self.audio.terminate()
        
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        
        print("Voice assistant stopped")


