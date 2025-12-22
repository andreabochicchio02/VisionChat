import time
import multiprocessing as mp
import psutil

from detected_object_module import object_detection_process
from voice_assistant_module import VoiceAssistant, configure_microphone_gain


def main():
    # Initialize audio hardware
    configure_microphone_gain()
    
    # Create inter-process communication queues
    detection_queue = mp.Queue()
    command_queue = mp.Queue()
    
    # Start object detection in separate process
    print("Launching object detection process...")
    detection_proc = mp.Process(
        target=object_detection_process,
        args=(detection_queue, command_queue)
    )
    detection_proc.start()
    
    # Pin detection process to specific CPU core for better performance
    try:
        proc = psutil.Process(detection_proc.pid)
        proc.cpu_affinity([1])      # Force detection process to use only CPU core 1
    except Exception as e:
        print(f"Could not set CPU affinity: {e}")
    
    # Allow detection process to initialize
    time.sleep(3)
    
    # Start voice assistant in main process
    assistant = VoiceAssistant(detection_queue, command_queue)
    
    try:
        assistant.start()
    finally:
        # Graceful shutdown
        print("\nShutting down...")
        
        # Stop detection process
        command_queue.put("STOP")
        detection_proc.join(timeout=5)      # Waits for the detection process to finish
        
        if detection_proc.is_alive():
            detection_proc.terminate()
            detection_proc.join()
        
        print("All processes stopped")


if __name__ == "__main__":
    main()