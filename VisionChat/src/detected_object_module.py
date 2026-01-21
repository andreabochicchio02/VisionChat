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
BLOB_SIZE = (320, 320)


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

class ObjectDetector:
    """Wraps the DNN and Picamera2 camera for frame capture + detection."""
    def __init__(self, language: str):
        self.net = None
        self.language = language
        self.classes = CLASSES.get(language)

        try:
            print("Initializing camera...")
            self.camera = Picamera2()

            config = self.camera.create_video_configuration(
                main={"size": FRAME_SIZE,},
                buffer_count=2  # Reduce buffer count to lower memory usage
            )

            self.camera.configure(config)

            self.camera.set_controls({
                "FrameRate": 10.0  # Limit to 10 FPS to reduce CPU load
            })

            self.camera.start()

            print("Camera started")
        except Exception as e:
            # Camera may not be present in some environments; keep object usable
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

    
    def capture_and_detect(self) -> Tuple[List[DetectedObject], np.ndarray]:
        """Capture one frame and run DNN detection.

        Returns a tuple `(detected_objects, drawn_frame_bytes)` where
        `detected_objects` is a list of DetectedObject instances and
        `drawn_frame` is a CV image (BGR).
        """

        # Capture and convert to OpenCV BGR
        frame = self.camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Prepare blob and run forward pass
        blob = cv2.dnn.blobFromImage(frame, 1.0 / 127.5, BLOB_SIZE, (127.5, 127.5, 127.5), True, False)
        self.net.setInput(blob)
        detections = self.net.forward()

        # SSD output: (1, 1, N, 7) -> [batch, class_id, confidence, x_min, y_min, x_max, y_max]
        detected_objects: List[DetectedObject] = []
        height, width = frame.shape[:2]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1])
                
                # Usa self.classes pre-caricata
                if class_id < len(self.classes) and self.classes[class_id] != "N/A":
                    # Extract bounding box coordinates
                    x_min, y_min, x_max, y_max = detections[0, 0, i, 3:7]
                    
                    class_name_str = self.classes[class_id]

                    detected_objects.append(DetectedObject(
                        class_name=class_name_str,
                        confidence=confidence * 100,
                        bounding_box=(x_min, y_min, x_max, y_max)
                    ))

                    # Draw bounding box
                    box_x = int(x_min * width)
                    box_y = int(y_min * height)
                    box_w = int((x_max - x_min) * width)
                    box_h = int((y_max - y_min) * height)

                    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
                    
                    # Label
                    label = f"{class_name_str}: {confidence*100:.0f}%"
                    cv2.putText(frame, label, (box_x, box_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detected_objects, frame
    
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

    - `detection_queue` receives lists of DetectedObject
    - `frame_queue` receives the latest JPEG bytes for streaming
    This function writes runtime logs to `log_camera.txt` in the same folder.
    """

    detector = ObjectDetector(language)
    f = open('log_camera.txt', 'w')
    sys.stdout = f

    try:
        detector.initialize()

        frame_count = 0
        start_time = time.time()
        
        while True:
            
            # Perform detection on current frame
            detections, frame_drawn = detector.capture_and_detect()
            detection_queue.put(detections)

            # STREAMING VIDEO HANDLING
            # Encode frame as JPG
            ret, buffer = cv2.imencode('.jpg', frame_drawn)
            if ret:
                # Empty the frame queue if it is full to avoid lag (we only want the last frame)
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