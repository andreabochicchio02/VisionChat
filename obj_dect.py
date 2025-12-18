from picamera2 import Picamera2
import cv2
import time
import numpy as np

# --- MODEL SETTINGS ---
model_weights = "models2/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
model_config = "models2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

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

print("Loading model...")
net = cv2.dnn.readNetFromTensorflow(model_weights, model_config)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model loaded.")

# --- CAMERA SETUP ---
picam = Picamera2()
picam.configure(picam.create_video_configuration(main={"size": (512, 384)})) 
picam.start()
time.sleep(1)
print("Camera started. Running detection...\n")

frame_count = 0
start_time = time.time()

try:
    while True:
        # Capture frame
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        (h, w) = frame.shape[:2]
        
        # Create blob and run detection
        blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(320, 320), 
                                     mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()
        
        # Process and print detections
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
        
        # Print frame info
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n--- Frame {frame_count} (FPS: {fps:.1f}) ---")
        if detected_objects:
            for obj in detected_objects:
                print(f"{obj['class']}: {obj['confidence']:.1f}% at box {obj['box']}")
        else:
            print("No objects detected")
        
        # Small delay to avoid overwhelming the terminal
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\nStopping detection...")
    picam.stop()
    print("Camera stopped.")