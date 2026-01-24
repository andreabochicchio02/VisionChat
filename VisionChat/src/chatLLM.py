import json
import requests
import os
import time
from typing import List, Tuple

from metrics_logger import create_llm_logger

# Constants IVAN
URL = "http://10.87.30.118:11434/api/generate"
MODEL = "llama3.2"

# Constants Andrea
# URL = "http://10.150.246.144:11434/api/generate"
# MODEL = "llama3.2:3b"

# Load prompts from JSON file
def load_prompts():
    """Load language-specific prompts from JSON file"""
    prompts_path = os.path.join(os.path.dirname(__file__), 'text.json')
    with open(prompts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {lang: data["text"][lang] for lang in data.get("text")}


PROMPTS = load_prompts()


class LLMClient:
    def __init__(self, lang: str, max_history: int = 3):
        """
        LLM client with conversation memory (last N question/answer pairs)
        """
        self.max_history = max_history
        self.history: List[Tuple[str, str]] = []    # (user_question, llm_answer)
        self.alert_object_list = set()              # Stored as a set of class names (strings)
        self.language = lang                        # en or it
        
        # Initialize metrics logger for LLM requests
        self.metrics_logger = create_llm_logger(".")
        
        # Track request counts for distinguishing first/second LLM requests
        self.request_count = 0


    def add_interaction(self, user_question: str, llm_answer: str) -> None:
        """
        Store a user question and LLM answer
        """
        if self.max_history == 0:
            return

        self.history.append((user_question, llm_answer))

        # Keep only last N interactions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_formatted_history(self) -> str:
        """Return formatted conversation history or 'no_history' when empty."""
        if not self.history:
            return "no_history"

        formatted = ""
        for user_q, llm_a in self.history:
            formatted += (
                f"User Question: {user_q}\n"
                f"LLM Answer: {llm_a}\n"
            )

        return formatted.strip()

    def add_alert_object(self, class_name: str) -> None:
        """Add an object class name to the alert list."""
        self.alert_object_list.add(class_name)

    def get_alert_objects(self) -> List[str]:
        """Return alert object class names as a list."""
        return list(self.alert_object_list)

    def detect_alert_request(self, user_text: str) -> dict:
        """Detect whether the user requests an alert and extract targets."""
        
        self.request_count += 1
        request_type = f"alert_detection_request_{self.request_count}"

        p = PROMPTS[self.language]['alert_detection']
        
        prompt = (
            f"{p['user_message']} \"{user_text}\"\n"
            f"{p['task']}"
            f"{p['user_message']} \"{user_text}\"\n"
        )

        payload = {"model": MODEL, "prompt": prompt, "stream": False, "format": "json"}
        
        # Log request start
        request_start = self.metrics_logger.log_llm_request_start(request_type, len(prompt))

        try:
            # Measure connection and send time
            send_start = time.time()
            response = requests.post(URL, json=payload, timeout=20)
            send_time_ms = (time.time() - send_start) * 1000
            
            response.raise_for_status()

            raw = response.json()
            generated_respose = raw['response']
            result_json = json.loads(generated_respose)
            print(result_json)
            
            # Log request completion
            self.metrics_logger.log_llm_request_end(
                request_start, 
                request_type, 
                generated_respose,
                len(generated_respose.split())  # Approximate token count
            )
            
            # Log network timing
            self.metrics_logger.log_llm_network_timing(0, send_time_ms, 0)

            return result_json


        except requests.exceptions.ConnectionError:
            error_msg = (
                "Error: unable to connect to Ollama server. "
                "Make sure Ollama is running with 'ollama serve'."
            )
            print(error_msg)
            self.metrics_logger.log_error("llm_connection_error", error_msg)
            return { "is_alert_request": False, "target_objects": []}

        except Exception as e:              # This also catches json.JSONDecodeError if the model returns invalid JSON
            print(f"An error occurred: {e}")
            self.metrics_logger.log_error("llm_request_error", str(e))
            return { "is_alert_request": False, "target_objects": []}



    def generate_response(self, objects: List, user_text: str) -> str:
        """Generate a response based on detected objects and conversation history"""
        
        self.request_count += 1
        request_type = f"response_generation_request_{self.request_count}"
        
        p = PROMPTS[self.language]['response_generation']

        # Build scene description
        scene_description = ""
        if objects:
            for i, obj in enumerate(objects, 1):
                position = obj.get_position_description(self.language)
                scene_description += f"{i}. {obj.class_name} - {p['position_label']} {position}\n"
        else:
            scene_description += p['no_objects']

        # Conversation history
        conversation_history = self.get_formatted_history()

        # Final prompt
        if self.max_history > 0:
            prompt = (
                f"{p['history_label']}\n"
                f"{conversation_history}\n\n"
                f"{p['objects_label']}\n"
                f"{scene_description}\n"
                f"{p['user_message_label']} {user_text}\n\n"
                f"{p['instruction']}\n"
                f"{p['user_message_label']} {user_text}\n"
            )
        else:
            # Skip history section when max_history is 0
            prompt = (
                f"{p['objects_label']}\n"
                f"{scene_description}\n"
                f"{p['user_message_label']} {user_text}\n\n"
                f"{p['instruction']}\n"
                f"{p['user_message_label']} {user_text}\n"
            )

        payload = {"model": MODEL, "prompt": prompt, "stream": True}

        print("\n\nPrompt", "=" * 25)
        print(prompt)
        print("=" * 25)

        full_response = ""
        token_count = 0
        first_token_logged = False
        
        # Log request start
        request_start = self.metrics_logger.log_llm_request_start(request_type, len(prompt))

        try:
            send_start = time.time()
            with requests.post(URL, json=payload, stream=True, timeout=20) as response:
                send_time_ms = (time.time() - send_start) * 1000
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue

                    chunk = json.loads(line)

                    if "response" in chunk:
                        token = chunk["response"]
                        token_count += 1
                        
                        # Log time to first token
                        if not first_token_logged:
                            self.metrics_logger.log_llm_first_token(request_start)
                            first_token_logged = True
                        
                        full_response += token
                        
                        # Yield each token for streaming to UI
                        yield token
                    if chunk.get('done', False):  # Stop if the 'done' flag is True
                        break

                print("\n")
                
                # Log request completion
                self.metrics_logger.log_llm_request_end(
                    request_start,
                    request_type,
                    full_response,
                    token_count
                )
                
                # Log network timing
                self.metrics_logger.log_llm_network_timing(0, send_time_ms, 0)

                # Save interaction after streaming is complete
                self.add_interaction(user_text, full_response)

        except requests.exceptions.ConnectionError:
            error_msg = (
                "Error: unable to connect to Ollama server. "
                "Make sure Ollama is running with 'ollama serve'."
            )
            print(error_msg)
            self.metrics_logger.log_error("llm_connection_error", error_msg)
            yield error_msg

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            print(error_msg)
            self.metrics_logger.log_error("llm_request_error", str(e))
            yield error_msg