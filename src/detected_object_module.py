import time
import queue
import multiprocessing as mp
import numpy as np
import cv2
import sys
from typing import List
from picamera2 import Picamera2


# Object detection model configuration 
MODEL_WEIGHTS = "models/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
MODEL_CONFIG = "models/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
CONFIDENCE_THRESHOLD = 0.65
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
        self.box = bounding_box

    def get_position_description(self) -> str:
        """Calculate the relative position of the object in the image"""
        start_x, start_y, end_x, end_y = self.box
        
        # Calculate object center
        center_x = (start_x + end_x) / 2
        center_y = (start_y + end_y) / 2
        
        # Determine horizontal position (assuming normalized coordinates 0-1)
        if center_x < 0.33:
            pos_h = "left"
        elif center_x < 0.67:
            pos_h = "center"
        else:
            pos_h = "right"
        
        # Determine vertical position
        if center_y < 0.33:
            pos_v = "top"
        elif center_y < 0.67:
            pos_v = "middle"
        else:
            pos_v = "bottom"
        
        return f"{pos_v} {pos_h}"

    def __repr__(self) -> str:
        return f"{self.class_name} ({self.confidence:.1f}%)"



class ObjectDetector:
    
    def __init__(self):
        self.net = None
        self.camera = None
        self.latest_detections: List[DetectedObject] = []
        self.alert_enabled = {}  # class_name -> bool, to manage alerts
        
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
                    # Extract bounding box coordinates
                    x_min, y_min, x_max, y_max = detections[0, 0, i, 3:7]
                    
                    detected_objects.append(DetectedObject(
                        class_name=CLASSES[class_id],
                        confidence=confidence * 100,
                        bounding_box=(x_min, y_min, x_max, y_max)
                    ))
        
        return detected_objects
    
    def cleanup(self) -> None:
        """Release camera resources"""
        if self.camera:
            self.camera.stop()
            print("Camera stopped")


def object_detection_process(detection_queue: mp.Queue, command_queue: mp.Queue, alert_queue: mp.Queue) -> None:
    """
    Main loop for the object detection process
    Continuously captures frames and updates detection results
    
    Args:
        command_queue: Queue to receive commands from main process
        detection_queue: Queue to send detection results to main process
    """
    detector = ObjectDetector()

    f = open('log_camera.txt', 'w')
    sys.stdout = f
    
    try:
        detector.initialize()
        
        # Log performance metrics
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Check for commands from main process
            try:
                command = command_queue.get_nowait()    # get a command from the queue without waiting
                # STOP command
                if command == "STOP":
                    break
                # Request to send current detections
                elif command == "GET_DETECTIONS":
                    detection_queue.put(detector.latest_detections.copy())
                # Add new alerts
                elif isinstance(command, tuple) and command[0] == "SET_ALERTS":
                    for cls in command[1]:
                        detector.alert_enabled[cls] = True
                    print(f"Added alerts: {command[1]}", flush=True)
                # Remove alerts
                elif isinstance(command, tuple) and command[0] == "REMOVE_ALERTS":
                    for cls in command[1]:
                        detector.alert_enabled.pop(cls, None)
                    print(f"Removed alerts: {command[1]}", flush=True)
            except queue.Empty:
                pass    # No command, continue
            
            # Perform detection on current frame
            detector.latest_detections = detector.capture_and_detect()

            #alert handling
            visible_classes = {obj.class_name for obj in detector.latest_detections}
            
            for cls, enabled in list(detector.alert_enabled.items()):
                # If object is visible and alert allowed
                if cls in visible_classes and enabled:
                    obj = next(o for o in detector.latest_detections if o.class_name == cls)
                    alert_queue.put({'object': obj, 'timestamp': time.time()})
                    
                    detector.alert_enabled[cls] = False
                    print(f"Sent alert for {cls}", flush=True)

                # If object not visible -> re-enable alert
                elif cls not in visible_classes:
                    if not detector.alert_enabled[cls]:
                        detector.alert_enabled[cls] = True
                        print(f"Re-enabled alert for {cls}", flush=True)

            # Log performance metrics
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"Frame {frame_count}, FPS: {fps:.1f}, Objects: {len(detector.latest_detections)}", flush=True)
            

            # Every 5 frames, print detailed detections
            if frame_count % 5 == 0:
                for i, obj in enumerate(detector.latest_detections, start=1):
                    print(
                        f"  Object {i} | Class: {obj.class_name}, "
                        f"Confidence: {obj.confidence:.2f}, "
                        f"Bounding Box: {obj.box}",
                        flush=True
                    )


            # Throttle frame rate to reduce CPU usage
            time.sleep(FRAME_DELAY)
    
    except Exception as e:
        # stdout on console
        sys.stdout = sys.__stdout__
        f.close()

        print(f"Error stopping camera {e}")
    finally:
        detector.cleanup()