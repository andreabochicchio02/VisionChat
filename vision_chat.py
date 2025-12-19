import argparse
import json
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


import numpy as np
import pyaudio
import cv2
import psutil
import queue
import multiprocessing as mp


from picamera2 import Picamera2
from vosk import Model, KaldiRecognizer

# Audio stream configuration 
RATE = 44100
CHUNK = 4096
MIC_CARD = "1"
MIC_CONTROL_NAME = "Mic"


# Object detection model configuration 
MODEL_WEIGHTS = "models2/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
MODEL_CONFIG = "models2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
CONFIDENCE_THRESHOLD = 0.55
FRAME_SIZE = (512, 384)
BLOB_SIZE = (320, 320)
FRAME_DELAY = 0.5  # TODO CHANGE IF NECESSARY


# Speech recognition and synthesis configuration
VOSK_MODEL_PATH = "models2/vosk-model-small-en-us-0.15"
ESPEAK_VOICE = "en-us"
ESPEAK_SPEED = "140"



# Class list (COCO labels used by this SSD model)
CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table",
    "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class DetectedObject:
    def __init__(self, class_name, confidence, bounding_box):
        self.class_name = class_name
        self.confidence = confidence
        self.box = bounding_box  # Qui assegniamo l'argomento 'bounding_box' alla variabile 'box'

    def __repr__(self) -> str:
        return f"{self.class_name} ({self.confidence:.1f}%)"




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





# ============================================================================
# Object Detection Process
# ============================================================================

class ObjectDetector:
    
    def __init__(self):
        self.net = None
        self.camera = None
        self.latest_detections: List[DetectedObject] = []
        
    def initialize(self) -> None:
        """Initialize the detection model and camera"""

        print("Loading DNN model...")

        self.net = cv2.dnn.readNetFromTensorflow(
            MODEL_WEIGHTS,
            MODEL_CONFIG
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print("Model loaded")
        

        print("Initializing camera...")
        self.camera = Picamera2()
        self.camera.configure(
            self.camera.create_video_configuration(
                main={"size": FRAME_SIZE}
            )
        )
        self.camera.start()
        time.sleep(1)  # Allow camera to warm up
        print("Camera started")
    
    def capture_and_detect(self) -> List[DetectedObject]:
        """
        Capture a frame and perform object detection
        """
        # Capture frame
        frame = self.camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # RGB -> BGR (OpenCV uses BGR)
        height, width = frame.shape[:2]
        
        # Prepare input blob for neural network
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 127.5,
            size=BLOB_SIZE,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False
        )
        
        # RUN DETECTION
        # SSD output shape: (1, 1, N, 7)
        # N = number of detected objects
        # Each detection:
        #       [0] batch_id, [1] class_id, [2] confidence,
        #       [3] x_min, [4] y_min, [5] x_max, [6] y_max (normalized)


        self.net.setInput(blob)
        detections = self.net.forward()
        
        # Process detection results
        detected_objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                
                # Validate class ID and skip N/A classes
                if class_id < len(CLASSES) and CLASSES[class_id] != "N/A":
                    # Extract and scale bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    start_x, start_y, end_x, end_y = box.astype("int")
                    
                    detected_objects.append(DetectedObject(
                        class_name=CLASSES[class_id],
                        confidence=confidence * 100,
                        bounding_box=(start_x, start_y, end_x, end_y)
                    ))
        
        return detected_objects
    
    def cleanup(self) -> None:
        """Release camera resources"""
        if self.camera:
            self.camera.stop()
            print("Camera stopped")


def object_detection_process(detection_queue: mp.Queue, command_queue: mp.Queue) -> None:
    """
    Main loop for the object detection process
    Continuously captures frames and updates detection results
    
    Args:
        command_queue: Queue to receive commands from main process
        detection_queue: Queue to send detection results to main process
    """
    detector = ObjectDetector()
    
    try:
        detector.initialize()
        
        # Log performance metrics
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Check for commands from main process
            try:
                command = command_queue.get_nowait()    # get a command from the queue without waiting
                if command == "STOP":
                    break
                elif command == "GET_DETECTIONS":
                    # Send current detection results
                    detection_queue.put(detector.latest_detections.copy())
            except queue.Empty:
                pass    # No command, continue
            
            # Perform detection on current frame
            detector.latest_detections = detector.capture_and_detect()
            
            # Log performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Frame {frame_count}, FPS: {fps:.1f}, Objects: {len(detector.latest_detections)}")
            
            # Throttle frame rate to reduce CPU usage
            time.sleep(FRAME_DELAY)
    
    except Exception as e:
        print(f"Error stopping camera {e}")
    finally:
        detector.cleanup()





# ============================================================================
# Speech Recognition
# ============================================================================

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




# ============================================================================
# Main Application
# ============================================================================

def main():
    # Initialize audio hardware
    configure_microphone_gain()
    
    # Create inter-process communication queues
    detection_queue = mp.Queue()
    command_queue = mp.Queue()
    
    # Start object detection in separate process
    print("Launching object detection process...")
    detection_proc = mp.Process(
        target=object_detection_process,
        args=(detection_queue, command_queue)
    )
    detection_proc.start()
    
    # Pin detection process to specific CPU core for better performance
    try:
        proc = psutil.Process(detection_proc.pid)
        proc.cpu_affinity([1])      # Force detection process to use only CPU core 1
    except Exception as e:
        print(f"Could not set CPU affinity: {e}")
    
    # Allow detection process to initialize
    time.sleep(3)
    
    # Start voice assistant in main process
    assistant = VoiceAssistant(detection_queue, command_queue)
    
    try:
        assistant.start()
    finally:
        # Graceful shutdown
        print("\nShutting down...")
        
        # Stop detection process
        command_queue.put("STOP")
        detection_proc.join(timeout=5)      # Waits for the detection process to finish
        
        if detection_proc.is_alive():
            detection_proc.terminate()
            detection_proc.join()
        
        print("All processes stopped")


if __name__ == "__main__":
    main()