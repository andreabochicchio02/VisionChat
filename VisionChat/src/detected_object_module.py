import time
import queue
import multiprocessing as mp
import cv2
import sys
import json
import os
import numpy as np
from typing import List, Tuple
from picamera2 import Picamera2


MODEL_WEIGHTS = "models/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
MODEL_CONFIG = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
CONFIDENCE_THRESHOLD = 0.50
FRAME_SIZE = (640, 480)  
BLOB_SIZE = (224, 224)   
JPEG_QUALITY = 90        # Lower quality for faster encoding
DETECTION_INTERVAL = 3   # Run detection every N frames


def load_json_data() -> dict:
    """Load localization JSON (`text.json`) from the module directory."""
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'text.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Critical error loading text.json: {e}")
        sys.exit(1)

# Carichiamo i dati globalmente una volta sola all'importazione del modulo
_JSON = load_json_data()
POSITIONS = _JSON['positions']
CLASSES = _JSON['classes']

class DetectedObject:
    """Small container representing one detected object."""
    
    def __init__(self, class_name: str, confidence: float, bounding_box: Tuple[float, float, float, float]):
        self.class_name = class_name
        self.confidence = confidence
        self.box = bounding_box  # (x_min, y_min, x_max, y_max) normalized

    def get_position_description(self, language: str) -> str:
        """Return a short localized position"""
        
        start_x, start_y, end_x, end_y = self.box
        
        # Calculate object center
        center_x = (start_x + end_x) / 2
        center_y = (start_y + end_y) / 2
        
        # Determine horizontal position index (0=Left, 1=Center, 2=Right)
        if center_x < 0.33: idx_h = 0
        elif center_x < 0.67: idx_h = 1
        else: idx_h = 2
        
        # Determine vertical position index (0=Top, 1=Middle, 2=Bottom)
        if center_y < 0.33: idx_v = 0
        elif center_y < 0.67: idx_v = 1
        else: idx_v = 2

        pos_data = POSITIONS.get(language)
        pos_v = pos_data['v'][idx_v]
        pos_h = pos_data['h'][idx_h]
        return f"{pos_v} {pos_h}"

    def __repr__(self) -> str:
        return f"{self.class_name} ({self.confidence:.1f}%)"


class MotionDetector:
    """
    Lightweight motion detector optimized for Raspberry Pi.
    Uses simple frame differencing with downscaled frames.
    """
    
    def __init__(self, threshold: int = 25, min_area_percent: float = 0.5, scale_factor: float = 0.25):
        """
        Args:
            threshold: Pixel difference threshold (0-255)
            min_area_percent: Minimum percentage of frame area to count as motion
            scale_factor: Downscale factor for processing (0.25 = 1/4 resolution)
        """
        self.threshold = threshold
        self.min_area_percent = min_area_percent
        self.scale_factor = scale_factor
        self.prev_gray = None
    
    def detect_motion(self, frame: np.ndarray) -> bool:
        """
        Detect motion using lightweight frame differencing.
        Returns True if significant motion is detected.
        """
        # Downscale for faster processing
        small = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, 
                          interpolation=cv2.INTER_NEAREST)
        
        # Convert to grayscale
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Light blur to reduce noise (small kernel for speed)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        
        # Compute absolute difference
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        
        # Binary threshold
        _, thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Count changed pixels
        motion_pixels = cv2.countNonZero(thresh)
        total_pixels = gray.shape[0] * gray.shape[1]
        motion_percent = (motion_pixels / total_pixels) * 100
        
        # Update previous frame
        self.prev_gray = gray
        
        return motion_percent > self.min_area_percent
    
    def reset(self) -> None:
        """Reset the motion detector state."""
        self.prev_gray = None


class ObjectDetector:
    """Wraps the DNN and Picamera2 camera for frame capture + detection."""
    def __init__(self, language: str):
        self.net = None
        self.language = language
        self.classes = CLASSES.get(language)
        self._last_detections: List[DetectedObject] = []  # Cache last detections

        try:
            print("Initializing camera...")
            self.camera = Picamera2()

            config = self.camera.create_video_configuration(
                main={"size": FRAME_SIZE,},
                buffer_count=2  # Reduce buffer count to lower memory usage
            )

            self.camera.configure(config)

            self.camera.set_controls({
                "FrameRate": 10.0
            })

            self.camera.start()

            print("Camera started")
        except Exception as e:
            print(f"Failed to start camera: {e}", flush=True)
            self.camera = None

    def initialize(self) -> None:
        """Initialize the detection model and camera"""

        print("Loading DNN model...")
        try:
            self.net = cv2.dnn.readNetFromTensorflow(MODEL_WEIGHTS, MODEL_CONFIG)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Model loaded")
        except Exception as e:
            print("Error loading model:", e)
            self.net = None

    def capture_frame(self) -> np.ndarray:
        """Capture one frame and convert to BGR."""
        frame = self.camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def detect_objects(self, frame: np.ndarray) -> List[DetectedObject]:
        """Run DNN detection on the given frame."""
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, BLOB_SIZE, (127.5, 127.5, 127.5), True, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        detected_objects: List[DetectedObject] = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                
                if class_id < len(self.classes) and self.classes[class_id] != "N/A":
                    x_min, y_min, x_max, y_max = detections[0, 0, i, 3:7]
                    class_name_str = self.classes[class_id]

                    detected_objects.append(DetectedObject(
                        class_name=class_name_str,
                        confidence=confidence * 100,
                        bounding_box=(x_min, y_min, x_max, y_max)
                    ))

        self._last_detections = detected_objects
        return detected_objects

    def draw_detections(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """Draw bounding boxes on the frame."""
        height, width = frame.shape[:2]

        for obj in detections:
            x_min, y_min, x_max, y_max = obj.box
            box_x = int(x_min * width)
            box_y = int(y_min * height)
            box_w = int((x_max - x_min) * width)
            box_h = int((y_max - y_min) * height)

            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)  # Thinner line
            label = f"{obj.class_name}: {obj.confidence:.0f}%"
            cv2.putText(frame, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def capture_and_detect(self) -> Tuple[List[DetectedObject], np.ndarray]:
        """Capture one frame and run DNN detection (legacy method)."""
        frame = self.capture_frame()
        detections = self.detect_objects(frame)
        frame = self.draw_detections(frame, detections)
        return detections, frame
    
    def cleanup(self) -> None:
        """Stop camera if running and free resources."""
        if getattr(self, 'camera', None):
            try:
                self.camera.stop()
            except Exception:
                pass
            print("Camera stopped")

def object_detection_process(detection_queue: mp.Queue, frame_queue: mp.Queue, language: str) -> None:
    """Process entrypoint: capture frames, detect objects, and stream frames.

    - `detection_queue` receives dicts with 'objects' and 'motion_detected' keys
    - `frame_queue` receives the latest JPEG bytes for streaming
    This function writes runtime logs to `log_camera.txt` in the same folder.
    """

    detector = ObjectDetector(language)
    motion_detector = MotionDetector(threshold=25, min_area_percent=0.5, scale_factor=0.25)
    f = open('log_camera.txt', 'w')
    sys.stdout = f

    # JPEG encoding parameters
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    try:
        detector.initialize()

        frame_count = 0
        start_time = time.time()
        
        while True:
            # Capture frame
            frame = detector.capture_frame()
            
            # Run motion detection every frame (lightweight)
            motion_detected = motion_detector.detect_motion(frame)
            
            # Run object detection only every N frames
            if frame_count % DETECTION_INTERVAL == 0:
                detections = detector.detect_objects(frame)
                # Send both object detections and motion status
                detection_queue.put({
                    'objects': detections,
                    'motion_detected': motion_detected
                })
            else:
                detections = detector._last_detections

            # Draw detections on frame
            frame_drawn = detector.draw_detections(frame, detections)

            # STREAMING VIDEO HANDLING
            ret, buffer = cv2.imencode('.jpg', frame_drawn, encode_params)
            if ret:
                if not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(buffer.tobytes())

            
            # Log performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0

            print(f"Frame {frame_count}, FPS: {fps:.1f}, Objects: {len(detections)}", flush=True)
            

            # Every 5 frames, print detailed detections
            if frame_count % 5 == 0:
                for i, obj in enumerate(detections, start=1):
                    print(f"  Object {i} | Class: {obj.class_name}, Confidence: {obj.confidence:.2f}, Bounding Box: {obj.box}", flush=True)

    except Exception as e:
        print(f"Error camera process: {e}")
    finally:
        detector.cleanup()

        # stdout on console
        sys.stdout = sys.__stdout__
        f.close()