import time
import cv2
import os
import gc  # IMPORTANTE: Garbage Collector interface
from ultralytics import YOLO
from picamera2 import Picamera2

# --- CONFIGURATION ---
PHOTO_WIDTH = 320  
PHOTO_HEIGHT = 240
# Increased confidence to reduce NMS workload (CPU load)
CONFIDENCE_THRESHOLD = 0.60 
MAX_DETECTIONS = 5

# --- 1. LOAD MODEL ---
print("Loading model...")
try:
    # Load NCNN model
    model = YOLO("yolov5nu_ncnn_model", task="detect")
    print("SUCCESS: YOLOv5nu NCNN model loaded.")
except Exception as e:
    print(f"Info: {e}. Loading standard PyTorch model...")
    model = YOLO("yolov5nu.pt", task="detect")

# --- 2. SETUP CAMERA ---
print("Initializing Camera...")
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (PHOTO_WIDTH, PHOTO_HEIGHT), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()

# --- 3. MAIN LOOP ---
print(f" SYSTEM READY - MEMORY OPTIMIZED MODE\n")

input("\nPress ENTER to start (Press CTRL+C to stop)...")

try:
    while True:
        loop_start = time.time()

        # 1. Capture
        frame = picam2.capture_array()
        
        if frame is None:
            continue

        # Color correction
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2. Inference
        # Increased confidence helps reduce CPU load (fewer boxes to process)
        results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, imgsz=320, max_det=MAX_DETECTIONS)
        
        # 3. Results
        elapsed = time.time() - loop_start
        print(f"\nProcessed in {elapsed:.2f}s")
        
        det_found = False
        for result in results:
            for box in result.boxes:
                det_found = True
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                print(f"DETECTED: {name.upper()} ({conf:.1%})")
        
        if not det_found:
            print("No objects detected.")
        
        print("-" * 30)

        # --- CRITICAL MEMORY OPTIMIZATION ---
        # Explicitly delete heavy variables
        del frame
        del results
        
        # Force Python to clear RAM immediately
        gc.collect()
        
        # CPU Cooldown (prevents thermal throttling)
        print("Cooling down CPU...")
        time.sleep(2.0) 

except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"\nCritical Error: {e}")
finally:
    picam2.stop()
    print("Camera stopped.")
