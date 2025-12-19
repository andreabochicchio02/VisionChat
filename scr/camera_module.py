import time
import queue
import multiprocessing as mp
from typing import List
import numpy as np
import cv2
from picamera2 import Picamera2


# Object detection model configuration 
MODEL_WEIGHTS = "models/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
MODEL_CONFIG = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
CONFIDENCE_THRESHOLD = 0.55
FRAME_SIZE = (512, 384)
BLOB_SIZE = (320, 320)
FRAME_DELAY = 0.5  # TODO CHANGE IF NECESSARY


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