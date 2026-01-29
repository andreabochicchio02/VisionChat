"""
Metrics logging module for VisionChat.
Provides process-safe logging for performance measurements.

Each processor writes to its own log file to avoid conflicts:
- log_camera_metrics.jsonl: Camera/detection process metrics
- log_voice_metrics.jsonl: Voice assistant process metrics
- log_llm_metrics.jsonl: LLM request metrics
"""

import json
import time
import os
import threading
from typing import Optional, Dict, Any
from datetime import datetime


class MetricsLogger:
    """
    Thread-safe metrics logger that writes JSON lines to a file.
    Each log entry contains a timestamp, event type, and metrics data.
    """
    
    def __init__(self, log_file: str, processor_name: str):
        """
        Initialize the metrics logger.
        
        Args:
            log_file: Path to the log file (will be created if not exists)
            processor_name: Name of the processor (for identification)
        """
        self.log_file = log_file
        self.processor_name = processor_name
        self._lock = threading.Lock()
        self._start_time = time.time()
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        # Write session start marker
        self._log_event("session_start", {
            "processor": processor_name,
            "session_id": self._session_id
        })
    
    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log an event with timestamp.
        
        Args:
            event_type: Type of the event (e.g., 'frame_capture', 'llm_request')
            data: Dictionary of metrics data
        """
        entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "event_type": event_type,
            "processor": self.processor_name,
            "session_id": self._session_id,
            "elapsed_since_start": time.time() - self._start_time,
            **data
        }
        
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
    
    # ========================
    # Camera Process Metrics
    # ========================
    
    def log_frame_capture(self, frame_count: int, capture_time_ms: float) -> None:
        """Log frame capture timing."""
        self._log_event("frame_capture", {
            "frame_count": frame_count,
            "capture_time_ms": capture_time_ms
        })
    
    def log_object_detection(self, frame_count: int, detection_time_ms: float, 
                             num_objects: int, objects_detected: list) -> None:
        """Log object detection inference timing."""
        self._log_event("object_detection", {
            "frame_count": frame_count,
            "inference_time_ms": detection_time_ms,
            "num_objects": num_objects,
            "objects_detected": objects_detected
        })
    
    def log_motion_detection(self, frame_count: int, motion_time_ms: float, 
                             motion_detected: bool) -> None:
        """Log motion detection timing."""
        self._log_event("motion_detection", {
            "frame_count": frame_count,
            "motion_detection_time_ms": motion_time_ms,
            "motion_detected": motion_detected
        })
    
    def log_frame_encoding(self, frame_count: int, encoding_time_ms: float) -> None:
        """Log JPEG encoding timing."""
        self._log_event("frame_encoding", {
            "frame_count": frame_count,
            "encoding_time_ms": encoding_time_ms
        })
    
    def log_fps(self, frame_count: int, fps: float, classification_fps: float) -> None:
        """Log FPS metrics."""
        self._log_event("fps_metrics", {
            "frame_count": frame_count,
            "fps": fps,
            "classification_fps": classification_fps
        })
    
    # ========================
    # Voice Assistant Metrics
    # ========================
    
    def log_speech_recognition_start(self) -> float:
        """Log start of speech recognition, returns start timestamp."""
        start_time = time.time()
        self._log_event("speech_recognition_start", {})
        return start_time
    
    def log_speech_recognition_end(self, start_time: float, recognized_text: str) -> None:
        """Log end of speech recognition with timing and word count."""
        duration_ms = (time.time() - start_time) * 1000
        word_count = len(recognized_text.split()) if recognized_text else 0
        
        self._log_event("speech_recognition_end", {
            "duration_ms": duration_ms,
            "word_count": word_count,
            "char_count": len(recognized_text),
            "text_preview": recognized_text[:100] if recognized_text else ""
        })
    
    def log_tts_start(self, text: str) -> float:
        """Log start of text-to-speech, returns start timestamp."""
        start_time = time.time()
        word_count = len(text.split()) if text else 0
        self._log_event("tts_start", {
            "word_count": word_count,
            "char_count": len(text)
        })
        return start_time
    
    def log_tts_end(self, start_time: float, text: str) -> None:
        """Log end of text-to-speech with timing."""
        duration_ms = (time.time() - start_time) * 1000
        word_count = len(text.split()) if text else 0
        
        self._log_event("tts_end", {
            "duration_ms": duration_ms,
            "word_count": word_count,
            "ms_per_word": duration_ms / word_count if word_count > 0 else 0
        })
    
    def log_interaction_start(self) -> float:
        """Log start of a complete interaction cycle, returns start timestamp."""
        start_time = time.time()
        self._log_event("interaction_start", {})
        return start_time
    
    def log_interaction_end(self, start_time: float, user_words: int, 
                           response_words: int, success: bool) -> None:
        """Log end of complete interaction with total timing."""
        duration_ms = (time.time() - start_time) * 1000
        total_words = user_words + response_words
        
        self._log_event("interaction_end", {
            "total_duration_ms": duration_ms,
            "user_word_count": user_words,
            "response_word_count": response_words,
            "total_words": total_words,
            "ms_per_word": duration_ms / total_words if total_words > 0 else 0,
            "success": success
        })
    
    # ========================
    # LLM Request Metrics
    # ========================
    
    def log_llm_request_start(self, request_type: str, prompt_length: int) -> float:
        """Log start of LLM request, returns start timestamp."""
        start_time = time.time()
        self._log_event("llm_request_start", {
            "request_type": request_type,
            "prompt_length": prompt_length,
            "prompt_word_count": len(prompt_length.split()) if isinstance(prompt_length, str) else prompt_length
        })
        return start_time
    
    def log_llm_first_token(self, start_time: float) -> None:
        """Log time to first token (TTFT)."""
        ttft_ms = (time.time() - start_time) * 1000
        self._log_event("llm_first_token", {
            "time_to_first_token_ms": ttft_ms
        })
    
    def log_llm_request_end(self, start_time: float, request_type: str,
                            response_text: str, token_count: int) -> None:
        """Log end of LLM request with full timing."""
        duration_ms = (time.time() - start_time) * 1000
        word_count = len(response_text.split()) if response_text else 0
        
        self._log_event("llm_request_end", {
            "request_type": request_type,
            "total_duration_ms": duration_ms,
            "response_word_count": word_count,
            "response_char_count": len(response_text),
            "token_count": token_count,
            "ms_per_token": duration_ms / token_count if token_count > 0 else 0,
            "ms_per_word": duration_ms / word_count if word_count > 0 else 0
        })
    
    def log_llm_network_timing(self, connection_time_ms: float, 
                               send_time_ms: float, receive_time_ms: float) -> None:
        """Log network timing for LLM request."""
        self._log_event("llm_network_timing", {
            "connection_time_ms": connection_time_ms,
            "send_time_ms": send_time_ms,
            "receive_time_ms": receive_time_ms,
            "total_network_time_ms": connection_time_ms + send_time_ms + receive_time_ms
        })
    
    # ========================
    # Utility Methods
    # ========================
    
    def log_error(self, error_type: str, error_message: str) -> None:
        """Log an error event."""
        self._log_event("error", {
            "error_type": error_type,
            "error_message": str(error_message)
        })
    
    def log_session_end(self) -> None:
        """Log session end marker."""
        self._log_event("session_end", {
            "total_session_duration_s": time.time() - self._start_time
        })


# Factory functions for creating loggers for each processor
def create_camera_logger(log_dir: str = ".") -> MetricsLogger:
    """Create a logger for the camera/detection process."""
    return MetricsLogger(
        os.path.join(log_dir, "log_camera_metrics.jsonl"),
        "camera_process"
    )


def create_voice_logger(log_dir: str = ".") -> MetricsLogger:
    """Create a logger for the voice assistant process."""
    return MetricsLogger(
        os.path.join(log_dir, "log_voice_metrics.jsonl"),
        "voice_assistant"
    )


def create_llm_logger(log_dir: str = ".") -> MetricsLogger:
    """Create a logger for the LLM client."""
    return MetricsLogger(
        os.path.join(log_dir, "log_llm_metrics.jsonl"),
        "llm_client"
    )
