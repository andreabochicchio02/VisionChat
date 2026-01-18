import time
import multiprocessing as mp
import threading
import psutil
import json
from flask import Flask, render_template, Response, jsonify

from queue import Queue

from detected_object_module import object_detection_process
from voice_assistant_module import VoiceAssistant

app = Flask(__name__)

# Create inter-process communication queues
detection_queue = mp.Queue()
frame_queue = mp.Queue(maxsize=2)  # Queue for video frames (max 2 to avoid latency)

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
        """
        Generator that yields JSON updates as newline-delimited JSON.
        Each update is a complete JSON object on its own line.
        """
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
                # Send as Server-Sent Event
                yield f"data: {json.dumps(notification)}\n\n"
            except:
                # Send heartbeat to keep connection alive
                yield f": Notification Error\n\n"
    return Response(generate(), mimetype='text/event-stream')


def init_system():
    """Initialize the object detection and voice assistant systems"""
    global assistant, detection_proc
    
    # Start object detection in separate process
    print("Launching object detection process...")
    detection_proc = mp.Process(
        target=object_detection_process,
        args=(detection_queue, frame_queue)
    )
    detection_proc.start()
    
    # Pin detection process to specific CPU core for better performance
    try:
        proc = psutil.Process(detection_proc.pid)
        proc.cpu_affinity([1])  # Force detection process to use only CPU core 1
    except Exception as e:
        print(f"Could not set CPU affinity: {e}")
        pass
    
    time.sleep(3)  # Warmup period
    
    # Start voice assistant in main process
    assistant = VoiceAssistant(detection_queue, ui_notification_queue)
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