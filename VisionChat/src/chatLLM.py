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

# Reusable session for connection pooling (keeps connections alive)
_session = requests.Session()

# Load prompts from JSON file
def load_prompts():
    """Load language-specific prompts from JSON file"""
    prompts_path = os.path.join(os.path.dirname(__file__), 'text.json')
    with open(prompts_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {lang: data["text"][lang] for lang in data.get("text")}


def warmup_model() -> bool:
    """
    Send a minimal warmup request to preload the model into memory.
    This eliminates cold-start latency for the first real user request.
    
    Returns:
        True if warmup succeeded, False otherwise
    """
    print("Warming up LLM model...")
    warmup_start = time.time()
    
    # Minimal prompt to load the model without generating much
    payload = {
        "model": MODEL,
        "prompt": "Hi",
        "stream": False,
        "options": {
            "num_predict": 1  # Generate only 1 token (minimal work)
        }
    }
    
    try:
        response = _session.post(URL, json=payload, timeout=60)
        response.raise_for_status()
        warmup_time = time.time() - warmup_start
        print(f"LLM model warmed up in {warmup_time:.2f}s")
        return True
    except requests.exceptions.ConnectionError:
        print("Warning: Could not warm up LLM - Ollama server not reachable")
        return False
    except Exception as e:
        print(f"Warning: LLM warmup failed: {e}")
        return False


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
        """
        DEPRECATED: Alert detection is now done in the unified query.
        This method is kept for backward compatibility but returns empty result.
        """
        return {"is_alert_request": False, "is_motion_request": False, "target_objects": []}

    def generate_unified_response(self, objects: List, user_text: str) -> str:
        """
        Single unified LLM query that handles both alert detection and response generation.
        This eliminates the need for two separate LLM calls.
        
        The response format is:
        - If alert request: yields JSON with alert info, then confirmation message
        - If normal query: yields the response tokens directly
        """
        self.request_count += 1
        request_type = f"unified_request_{self.request_count}"
        
        p = PROMPTS[self.language]['response_generation']
        
        # Build scene description
        scene_description = ""
        if objects:
            for i, obj in enumerate(objects, 1):
                position = obj.get_position_description(self.language)
                scene_description += f"{i}. {obj.class_name} - {p['position_label']} {position}\n"
        else:
            scene_description += p['no_objects']

        # Get unified prompt from JSON
        unified_p = PROMPTS[self.language].get('unified_query', {})
        
        # Build unified prompt that handles both alert detection and response
        if self.language == "en":
            prompt = f"""You are a helpful visual assistant. Analyze the user's message and respond appropriately.

OBJECTS CURRENTLY VISIBLE:
{scene_description}

USER MESSAGE: "{user_text}"

INSTRUCTIONS:
1. FIRST, determine if the user is requesting a FUTURE notification/alert:
   - Alert keywords: "notify me", "alert me", "tell me when", "let me know", "warn me"
   - Motion keywords: "movement", "motion", "something moves"
   
2. IF this is an alert request:
   - Start your response with: [ALERT]
   - Then list any specific objects mentioned (e.g., [OBJECTS: dog, cat])
   - If asking for motion detection, add: [MOTION]
   - End with a brief confirmation like "I will notify you when I see that."

3. IF this is a normal question/conversation:
   - Answer directly and briefly
   - Only mention visible objects if the user asks about them
   - Use your general knowledge for other questions

RESPOND NOW:"""
        else:  # Italian
            prompt = f"""Sei un assistente visivo. Analizza il messaggio dell'utente e rispondi appropriatamente.

OGGETTI ATTUALMENTE VISIBILI:
{scene_description}

MESSAGGIO UTENTE: "{user_text}"

ISTRUZIONI:
1. PRIMA, determina se l'utente sta chiedendo una notifica/avviso FUTURO:
   - Parole chiave avviso: "avvisami", "notificami", "dimmi quando", "fammi sapere", "avvertimi"
   - Parole chiave movimento: "movimento", "si muove", "qualcosa si muove"
   
2. SE è una richiesta di avviso:
   - Inizia la risposta con: [ALERT]
   - Poi elenca gli oggetti specifici menzionati (es., [OBJECTS: cane, gatto])
   - Se chiede rilevamento movimento, aggiungi: [MOTION]
   - Termina con una breve conferma come "Ti avviserò quando lo vedrò."

3. SE è una domanda/conversazione normale:
   - Rispondi direttamente e brevemente
   - Menziona gli oggetti visibili solo se l'utente li chiede
   - Usa la tua conoscenza generale per altre domande

RISPONDI ORA:"""

        payload = {"model": MODEL, "prompt": prompt, "stream": True}

        print("\n\nUnified Prompt", "=" * 25)
        print(prompt)
        print("=" * 25)

        full_response = ""
        token_count = 0
        first_token_logged = False
        
        # Log request start
        request_start = self.metrics_logger.log_llm_request_start(request_type, len(prompt))

        try:
            send_start = time.time()
            with _session.post(URL, json=payload, stream=True, timeout=30) as response:
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
                    if chunk.get('done', False):
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

    def parse_alert_from_response(self, response_text: str) -> dict:
        """
        Parse the unified response to extract alert information.
        
        Returns:
            dict with keys: is_alert_request, is_motion_request, target_objects
        """
        result = {
            "is_alert_request": False,
            "is_motion_request": False,
            "target_objects": []
        }
        
        # Check for alert marker
        if "[ALERT]" in response_text.upper():
            result["is_alert_request"] = True
            
            # Check for motion marker
            if "[MOTION]" in response_text.upper():
                result["is_motion_request"] = True
            
            # Extract objects from [OBJECTS: ...] pattern
            import re
            objects_match = re.search(r'\[OBJECTS?:\s*([^\]]+)\]', response_text, re.IGNORECASE)
            if objects_match:
                objects_str = objects_match.group(1)
                # Split by comma and clean up
                objects = [obj.strip().lower() for obj in objects_str.split(',')]
                result["target_objects"] = [obj for obj in objects if obj]
        
        return result



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
            # Using session for connection pooling (reduces latency on subsequent requests)
            send_start = time.time()
            with _session.post(URL, json=payload, stream=True, timeout=20) as response:
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