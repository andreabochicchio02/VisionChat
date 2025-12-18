import subprocess
import pyaudio
import numpy as np
import threading
import queue
import time
from vosk import Model, KaldiRecognizer
import json
import multiprocessing as mp
from picamera2 import Picamera2
import cv2
import psutil

# Increase microphone gain
subprocess.run(["amixer", "-c", "1", "set", "Mic", "100%"], check=False)

# Audio configuration
RATE = 44100
CHUNK = 4096

# Model settings
MODEL_WEIGHTS = "models2/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
MODEL_CONFIG = "models2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

CLASSES = [
    "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet",
    "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def speak(text: str):
    """Text-to-speech using espeak"""
    subprocess.run(
        ["espeak", "-v", "en-us", "-s", "140", text], 
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


def object_detection_process(detection_queue, command_queue):
    """
    Separate process for object detection
    Continuously captures frames and detects objects
    Sends latest detection results when requested
    """
    print("🎥 [Detection Process] Loading model...")
    net = cv2.dnn.readNetFromTensorflow(MODEL_WEIGHTS, MODEL_CONFIG)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("✅ [Detection Process] Model loaded.")
    
    print("📷 [Detection Process] Starting camera...")
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"size": (512, 384)}))
    picam.start()
    time.sleep(1)
    print("✅ [Detection Process] Camera started.")
    
    latest_detections = []
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Check for commands from main process
            try:
                cmd = command_queue.get_nowait()
                if cmd == "STOP":
                    break
                elif cmd == "GET_DETECTIONS":
                    # Send latest detections to main process
                    detection_queue.put(latest_detections.copy())
            except queue.Empty:
                pass
            
            # Capture and process frame
            frame = picam.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            (h, w) = frame.shape[:2]
            
            # Create blob and run detection
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(320, 320),
                                        mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()
            
            # Process detections
            detected_objects = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.55:
                    idx = int(detections[0, 0, i, 1])
                    if idx < len(CLASSES) and CLASSES[idx] != "N/A":
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        detected_objects.append({
                            'class': CLASSES[idx],
                            'confidence': confidence * 100,
                            'box': (startX, startY, endX, endY)
                        })
            
            latest_detections = detected_objects
            
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"📊 [Detection Process] Frame {frame_count}, FPS: {fps:.1f}, Objects: {len(latest_detections)}")
        
            time.sleep(0.1)  # Small delay to reduce CPU usage
            
    except Exception as e:
        print(f"❌ [Detection Process] Error: {e}")
    finally:
        picam.stop()
        print("✅ [Detection Process] Camera stopped.")


class StreamingSpeechRecognizer:
    def __init__(self, detection_queue, command_queue, model_path="/home/studenti/vosk-model-small-en-us-0.15"):
        print("🧠 Loading Vosk model...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, RATE)
        self.recognizer.SetWords(True)
        
        self.audio = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.running = False
        self.speech_detected = False
        self.is_speaking = False
        
        # Queues for communication with detection process
        self.detection_queue = detection_queue
        self.command_queue = command_queue
        
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
        if not self.is_speaking:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    def _process_audio(self):
        """Thread that processes audio"""
        while self.running:
            if self.is_speaking:
                time.sleep(0.1)
                continue
                
            try:
                data = self.audio_queue.get(timeout=0.1)
                
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "")
                    
                    if text:
                        print(f"\n✅ PHRASE: {text}")
                        self.send_to_llm(text)
                        self.speech_detected = False
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    text = partial.get("partial", "")
                    if text:
                        if not self.speech_detected:
                            self.speech_detected = True
                            print(f"\n🎯 Speech detected!")
                        print(f"\r🗣️  {text}", end="", flush=True)
                    
            except queue.Empty:
                continue
    
    def send_to_llm(self, text):
        """Send text to LLM with current object detection results"""
        print("🤖 Processing with LLM...")
        
        # Request latest detections from detection process
        self.command_queue.put("GET_DETECTIONS")
        
        # Wait for detections (with timeout)
        try:
            detected_objects = self.detection_queue.get(timeout=2.0)
        except queue.Empty:
            detected_objects = []
            print("⚠️  No detection data received")
        
        # Format detected objects for LLM
        if detected_objects:
            object_list = [f"{obj['class']} ({obj['confidence']:.1f}%)" for obj in detected_objects]
            objects_str = ", ".join(object_list)
            print(f"👁️  Detected objects: {objects_str}")
        else:
            objects_str = "no objects detected"
            print(f"👁️  No objects detected")
        
        # TODO: Replace with actual LLM call
        # For now, generate a simple response based on detected objects
        if detected_objects:
            response = f"I can see {len(detected_objects)} object"
            if len(detected_objects) > 1:
                response += "s"
            response += f": {objects_str}"
        else:
            response = "I don't see any objects in the current view"
        
        print(f"💬 Response: {response}")
        print("🔊 Speaking...")
        
        # Set flag to pause audio processing
        self.is_speaking = True
        
        # Clear audio queue
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        speak(response)
        
        # Reset flag to resume audio processing
        self.is_speaking = False
        
        print("✅ Speech complete\n")
    
    def start(self):
        """Start streaming"""
        print("🎙️  Starting audio stream...")
        
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
    # Create queues for inter-process communication
    detection_queue = mp.Queue()
    command_queue = mp.Queue()
    
    # Start object detection process
    print("🚀 Starting object detection process...")
    detection_proc = mp.Process(target=object_detection_process, args=(detection_queue, command_queue))
    detection_proc.start()

    p = psutil.Process(detection_proc.pid)
    p.cpu_affinity([1])   # Force detection process to use only CPU core 1
    
    # Give detection process time to initialize
    time.sleep(3)
    
    # Start speech recognition in main process
    print("🚀 Starting speech recognition...")
    recognizer = StreamingSpeechRecognizer(detection_queue, command_queue)
    
    try:
        recognizer.start()
    finally:
        # Clean shutdown
        print("\n🛑 Shutting down...")
        command_queue.put("STOP")
        detection_proc.join(timeout=5)
        if detection_proc.is_alive():
            print("⚠️  Force terminating detection process...")
            detection_proc.terminate()
            detection_proc.join()
        print("✅ All processes stopped")