from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import time
import numpy as np

app = Flask(__name__)

# --- CONFIGURAZIONE OBJECT DETECTION ---
# Percorsi ai file del modello (assicurati di averli nella stessa cartella)
prototxt_path = "models/SSD_MobileNet_prototxt.txt"
model_path = "models/SSD_MobileNet.caffemodel"

# Classi supportate da MobileNet SSD
CLASSES = ["sfondo", "aereo", "bicicletta", "uccello", "barca",
           "bottiglia", "autobus", "auto", "gatto", "sedia", "mucca", "tavolo",
           "cane", "cavallo", "moto", "persona", "pianta",
           "pecora", "divano", "treno", "monitor"]

# Colori casuali per ogni classe
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Caricamento modello...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
print("Modello caricato.")

# --- CONFIGURAZIONE CAMERA ---
picam = Picamera2()
picam.configure(picam.create_video_configuration(main={"size": (640, 480)}))
picam.start()
time.sleep(1)

def generate_stream():
    frame_count = 0
    detections = None # Memorizza le rilevazioni per disegnarle anche nei frame saltati
    
    while True:
        # Capture frame (Picamera2 cattura RGB, OpenCV vuole BGR)
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        (h, w) = frame.shape[:2]

        # --- LOGICA DI RILEVAMENTO ---
        # Eseguiamo il rilevamento pesante solo ogni 5 frame per non bloccare il video
        if frame_count % 5 == 0:
            # Prepara l'immagine per la rete neurale (resize a 300x300, normalizzazione)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, 
                                         (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

        # Disegniamo i risultati (usiamo l'ultimo 'detections' calcolato)
        if detections is not None:
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                # Filtra rilevazioni deboli (soglia 0.5 = 50%)
                if confidence > 0.5:
                    idx = int(detections[0, 0, i, 1])
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Disegna rettangolo e etichetta
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # Incrementa contatore
        frame_count += 1

        # Encode as JPEG
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() + 
            b"\r\n"
        )

@app.route("/video")
def video():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def index():
    return "<h1>Live Streaming con AI</h1><img src='/video'>"

if __name__ == "__main__":
    # threaded=True è essenziale per Flask + Camera
    app.run(host="0.0.0.0", port=5000, threaded=True)