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
        self.listening = False
        
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
        Queues audio data only when listening is active
        """

        if self.listening:  
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _record_and_recognize(self) -> None:
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
    
    def _handle_user_input(self, input: str) -> None:
        """
        Process recognized input (text) and generate response
        """
        
        # Request latest object detections
        self.command_queue.put("GET_DETECTIONS")
        
        # Wait for detection results
        try:
            detected_objects: List[DetectedObject] = self.detection_queue.get(timeout=2.0)
        except queue.Empty:
            detected_objects = []
            print("Detection data not available")
        
        # Generate response
        response = self._generate_response(detected_objects, input)
        
        self._speak_response(response)
    
    def _generate_response(self, objects: List[DetectedObject], user_text: str) -> str:
        """
        Generate a response based on detected objects
        """

        # Build prompt
        scene_description = "OBJECTS DETECTED BY CAMERA:\n"

        if objects:
            for i, obj in enumerate(objects, 1):
                position = obj.get_position_description()
                scene_description += f"{i}. {obj.class_name} - Position: {position}\n"
        else:
            scene_description += "No objects detected.\n"

        
        prompt = (
                "OBJECTS DETECTED BY CAMERA:\n"
                f"{scene_description}\n"
                f"USER QUESTION: {user_text}\n"
                "INSTRUCTION: Answer the question taking into account "
                "the objects detected by the camera and their positions.")


        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True 
        }

        print("\nREQUEST", "="*25)
        print(prompt)
        print("\nRESPONSE", "="*25)

        full_response = ""
    
        try:
            # Execute streaming request
            with requests.post(URL, json=payload, stream=True, timeout=30) as response:
                response.raise_for_status()
                
                # Process streaming response line by line
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        
                        if 'response' in chunk:
                            # Print in streaming
                            sys.stdout.write(chunk['response'])
                            sys.stdout.flush()
                            
                            # Accumulate response text
                            full_response += chunk['response']
                
                print("\n")
                
                # Return the complete response
                return full_response
    
        except requests.exceptions.ConnectionError:
            error_msg = "Error: unable to connect to Ollama server. Make sure Ollama is running with the command 'ollama serve'."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"An error occurred: {e}"
            print(error_msg)
            return error_msg

    
    def _speak_response(self, text: str) -> None:
        """
        Speak a response and pause audio processing during speech
        
        Args:
            text: Text to convert to speech
        """
        
        # Perform text-to-speech
        text_to_speech(text)
        
        print("Speech complete")
    
    def start(self) -> None:
        """Start the voice assistant
        
        # EXECUTION   #TODO Da Cambiare
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
        
        print("Voice Assistant active!")

        try:
            while self.running:
                input("Press ENTER to speak... ")

                recognized_text = self._record_and_recognize()

                # If text was recognized, process the request
                if recognized_text:
                    print(f"\nRecognized Input: {recognized_text}")
                    self._handle_user_input(recognized_text)
                else:
                    print("No voice input recognized. Try again")

                print("="*50, "\n")
        
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


