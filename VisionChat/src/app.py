import time
import multiprocessing as mp
import json
from flask import Flask, render_template, Response, jsonify

from queue import Queue

from detected_object_module import object_detection_process
from voice_assistant_module import VoiceAssistant
from chatLLM import warmup_model

app = Flask(__name__)

# Create inter-process communication queues
detection_queue = mp.Queue()            # Queue for object detected
frame_queue = mp.Queue()                # Queue for video frames

ui_notification_queue = Queue() # Queue for UI notifications

assistant = None
detection_proc = None


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

def gen_frames():
    """Generator function for video streaming"""
    while True:
        try:
            # Get JPEG frame from queue
            frame_data = frame_queue.get(timeout=1.0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        except:
            # If no frames (e.g. startup or error), send placeholder or continue
            continue

@app.route('/video_feed')
def video_feed():
    """Stream video feed to the UI"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/listen', methods=['POST'])
def listen():
    """
    Returns a streaming response that sends updates as they happen:
    1. User text as soon as it's recognized
    2. Assistant response after LLM processing
    """
    if not assistant:
        return jsonify({"error": "Assistant not initialized"}), 500
    
    def generate():
        """Yield newline-delimited JSON updates from assistant."""
        try:
            for update in assistant.process_single_interaction_streaming():
                # Send each update as a separate JSON line
                yield json.dumps(update) + '\n'
        except Exception as e:
            # Send error if something goes wrong
            yield json.dumps({"error": str(e)}) + '\n'
    
    # Return streaming response with newline-delimited JSON
    return Response(generate(), mimetype='application/json')


@app.route('/notifications')
def notifications():
    """
    Server-Sent Events endpoint for real-time notifications.
    Sends alerts from the alert_listener thread to the UI.
    """
    def generate():
        """Generator that yields Server-Sent Events"""
        while True:
            try:
                # Get notification from queue (blocking with timeout)
                notification = ui_notification_queue.get(timeout=1.0)

                msg = str(notification.get("notification", ""))
                msg = msg.replace("\n", " ").strip()

                payload = json.dumps({"notification": msg}, ensure_ascii=False)

                yield f"data: {payload}\n\n"

            except:
                # Send heartbeat to keep connection alive
                yield ": keep-alive\n\n"
    return Response(generate(), mimetype='text/event-stream', headers={"Cache-Control": "no-cache","X-Accel-Buffering": "no"})


def init_system():
    """Initialize the object detection and voice assistant systems"""
    global assistant, detection_proc

    lang = input("Select language (it/en): ").strip().lower()
    if lang not in ['it', 'en']:
        print("Invalid language, defaulting to English")
        lang = 'it'
    
    # Warm up the LLM model to eliminate cold-start latency
    # This preloads the model into GPU/CPU memory before user interaction
    warmup_model()
    
    # Start object detection in separate process
    print("Launching object detection process...")
    detection_proc = mp.Process(
        target=object_detection_process,
        args=(detection_queue, frame_queue, lang)
    )
    detection_proc.start()
    
    time.sleep(1)  # Warmup period
    
    # Start voice assistant in main process
    assistant = VoiceAssistant(detection_queue, ui_notification_queue, lang)
    assistant.start_background_services()
    print("System Ready.")


if __name__ == "__main__":
    try:
        init_system()
        # Start Flask (blocks the main thread)
        # host='0.0.0.0' allows access from other devices on the network
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
    finally:
        print("\nShutting down...")
        if assistant:
            assistant.stop()
        if detection_proc:
            detection_proc.terminate()
            detection_proc.join()