from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time
import numpy as np

app = Flask(__name__)

# --- CONFIGURAZIONE MODELLO (SSD MobileNet V3 Large) ---
model_weights = "models2/ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
model_config = "models2/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# --- LISTA CLASSI CORRETTA (91 Indici per TensorFlow) ---
# Nota: "N/A" sono i segnaposto per gli indici che il modello salta.
# NON RIMUOVERE I "N/A"! Servono a mantenere l'allineamento.
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

print("Loading optimized V3 model...")
net = cv2.dnn.readNetFromTensorflow(model_weights, model_config)
print("Model loaded.")

# --- CAMERA ---
picam = Picamera2()
picam.configure(picam.create_video_configuration(main={"size": (640, 480)}))
picam.start()
time.sleep(1)

def generate_stream():
    frame_count = 0
    detections = None
    
    while True:
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        (h, w) = frame.shape[:2]

        # --- LOGICA DI RILEVAMENTO ---
        if frame_count % 10 == 0:
            # CORREZIONE FONDAMENTALE:
            # 1. size=(320, 320): dimensione specifica per V3
            # 2. scalefactor=1.0 / 127.5: riduce i numeri grandi
            # 3. mean=(127.5, 127.5, 127.5): centra i valori sullo zero
            # 4. swapRB=True: converte BGR in RGB per il modello
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0/127.5, size=(320, 320), 
                                         mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()

        if detections is not None:
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Ho alzato leggermente la confidenza a 0.55 per ridurre falsi positivi
                if confidence > 0.55:
                    idx = int(detections[0, 0, i, 1])
                    
                    # Controllo di sicurezza sull'indice
                    if idx < len(CLASSES) and CLASSES[idx] != "N/A":
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        # Clipping per evitare crash se il box esce dallo schermo
                        startX = max(0, startX)
                        startY = max(0, startY)
                        endX = min(w, endX)
                        endY = min(h, endY)

                        label = "{}: {:.0f}%".format(CLASSES[idx], confidence * 100)
                        cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                        
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        frame_count += 1
        
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret: continue

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")

@app.route("/video")
def video():
    return Response(generate_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return "<h1>AI V3 Corrected</h1><img src='/video'>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)