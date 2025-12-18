from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time
import numpy as np
import threading

app = Flask(__name__)

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

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")
net = cv2.dnn.readNetFromTensorflow(model_weights, model_config)
# Optional: Try to use OpenCV optimized backend (helps on some Pi OS versions)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Model loaded.")

# --- SHARED VARIABLES (Thread Safety) ---
# These variables are shared between the Video Thread and the AI Thread
frame_to_process = None
latest_detections = None
lock = threading.Lock() # Ensures threads don't clash when writing variables

# --- AI WORKER THREAD ---
# This function runs in the background constantly
def ai_worker():
    global frame_to_process, latest_detections
    
    while True:
        # 1. Get the latest frame safely
        current_frame = None
        with lock:
            if frame_to_process is not None:
                current_frame = frame_to_process.copy()
        
        # 2. If we have a frame, run AI
        if current_frame is not None:
            blob = cv2.dnn.blobFromImage(current_frame, scalefactor=1.0/127.5, size=(320, 320), 
                                         mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()
            
            # 3. Update the shared results safely
            with lock:
                latest_detections = detections
        else:
            time.sleep(0.01) # Sleep if no frame yet to save CPU

# Start the AI thread immediately
t = threading.Thread(target=ai_worker)
t.daemon = True # Kills thread when app closes
t.start()


# --- CAMERA SETUP ---
picam = Picamera2()
# Reduced resolution slightly to 512x384 for better FPS on Pi 3
picam.configure(picam.create_video_configuration(main={"size": (512, 384)})) 
picam.start()
time.sleep(1)

def generate_stream():
    global frame_to_process
    
    while True:
        # Capture frame
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        (h, w) = frame.shape[:2]

        # --- SEND TO AI THREAD ---
        # We don't wait for AI here. We just update the "job" for the worker.
        with lock:
            frame_to_process = frame

        # --- DRAW LATEST KNOWN DETECTIONS ---
        # Get the last results calculated by the background thread
        current_detections = None
        with lock:
            current_detections = latest_detections

        if current_detections is not None:
            for i in range(current_detections.shape[2]):
                confidence = current_detections[0, 0, i, 2]

                if confidence > 0.55:
                    idx = int(current_detections[0, 0, i, 1])
                    if idx < len(CLASSES) and CLASSES[idx] != "N/A":
                        box = current_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        
                        # Clipping
                        startX, startY = max(0, startX), max(0, startY)
                        endX, endY = min(w, endX), min(h, endY)

                        label = "{}: {:.0f}%".format(CLASSES[idx], confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # --- OPTIMIZED ENCODING ---
        # Quality 80 is lighter than default (95) and looks almost the same
        ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret: continue

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return "<h1>AI Turbo Stream</h1><img src='/video'>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)