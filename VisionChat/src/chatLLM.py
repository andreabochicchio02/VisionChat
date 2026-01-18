import requests
import json
import sys
from typing import List, Tuple

# Constants
URL = "http://10.150.246.144:11434/api/generate"
MODEL = "llama3.2:3b"


class LLMClient:
    def __init__(self, max_history: int = 5):
        """
        LLM client with conversation memory (last N question/answer pairs)
        """
        self.max_history = max_history
        self.history: List[Tuple[str, str]] = []    # (user_question, llm_answer)
        self.alert_object_list = set()              # Stored as a set of class names (strings)


    def add_interaction(self, user_question: str, llm_answer: str) -> None:
        """
        Store a user question and LLM answer
        """
        self.history.append((user_question, llm_answer))

        # Keep only last N interactions
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_formatted_history(self) -> str:
        """
        Return conversation history formatted as:
        User Question: ...
        LLM Answer: ...
        """
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
        """Add an object class to the alert list"""
        self.alert_object_list.add(class_name)

    def get_alert_objects(self) -> list:
        """Return current alert object classes as a list"""
        return list(self.alert_object_list)

    def detect_alert_request(self, user_text: str) -> dict:
        """
        Detect if user is requesting an alert and extract target objects.
        Returns: dict with 'is_alert_request', 'response', and 'target_objects'
        """

        prompt = (
            f"USER MESSAGE: \"{user_text}\"\n"

            "TASK:\n"
            "Classify whether the user is asking for a FUTURE notification when objects appear.\n\n"

            "DECISION RULES (FOLLOW ALL):\n"
            "1. Set \"is_alert_request\" to true ONLY if the user explicitly asks for a future notification.\n"
            "2. The message MUST contain clear notification verbs such as:\n"
            "   'notify me', 'alert me', 'tell me when'.\n"
            "3. Questions about the current scene (e.g. 'what do you see?', 'what is there?') are NOT alert requests.\n"
            "4. If there is any doubt, set \"is_alert_request\" to false.\n"
            "5. DO NOT guess or invent objects.\n"
            "6. Extract target objects ONLY if they are explicitly mentioned in the user message.\n\n"

            "OUTPUT FORMAT (JSON ONLY):\n"
            "{\n"
            '  "is_alert_request": true or false,\n'
            '  "target_objects": ["object1", "object2"]\n'
            "}\n\n"

            "EXAMPLES:\n"
            "User: \"What can you see?\"\n"
            "Output:\n"
            "{ \"is_alert_request\": false, \"target_objects\": [] }\n\n"

            "User: \"Notify me when you see a dog\"\n"
            "Output:\n"
            "{ \"is_alert_request\": true, \"target_objects\": [\"dog\"] }\n\n"

            "User: \"Alert me if a person appears\"\n"
            "Output:\n"
            "{ \"is_alert_request\": true, \"target_objects\": [\"person\"] }\n\n"

            "User: \"Do you see a cat?\"\n"
            "Output:\n"
            "{ \"is_alert_request\": false, \"target_objects\": [] }\n\n"

            "Respond ONLY with valid JSON. No explanations.\n"
            f"USER MESSAGE: \"{user_text}\"\n"
        )


        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post(URL, json=payload, timeout=15)
            response.raise_for_status()

            raw = response.json()
            generated_respose = raw['response']
            result_json = json.loads(generated_respose)
            print(result_json)

            return result_json


        except requests.exceptions.ConnectionError:
            print(
                "Error: unable to connect to Ollama server. "
                "Make sure Ollama is running with 'ollama serve'."
            )
            return { "is_alert_request": False, "target_objects": []}

        except Exception as e:              # This also catches json.JSONDecodeError if the model returns invalid JSON
            print(f"An error occurred: {e}")
            return { "is_alert_request": False, "target_objects": []}



    def generate_response(self, objects: List, user_text: str) -> str:
        """
        Generate a response based on detected objects and conversation history
        """

        # Build scene description
        scene_description = ""
        if objects:
            for i, obj in enumerate(objects, 1):
                position = obj.get_position_description()
                scene_description += f"{i}. {obj.class_name} - Position: {position}\n"
        else:
            scene_description += "No objects detected.\n"

        # Conversation history
        conversation_history = self.get_formatted_history()

        # Final prompt
        prompt = (
            "CONVERSATION HISTORY:\n"
            f"{conversation_history}\n\n"
            "OBJECTS DETECTED BY CAMERA:\n"
            f"{scene_description}\n\n"
            f"USER MESSAGE: {user_text}\n"
            "INSTRUCTION: Provide a brief, focused answer. Answer the question taking into account "
            "the detected objects, their positions, and the previous conversation."
        )

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True
        }

        print("Prompt", "=" * 25)
        print(prompt)
        print("=" * 25)

        full_response = ""

        try:
            with requests.post(URL, json=payload, stream=True, timeout=30) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if "response" in chunk:
                        sys.stdout.write(chunk["response"])
                        sys.stdout.flush()
                        full_response += chunk["response"]

                print("\n")

                # Save interaction
                self.add_interaction(user_text, full_response)

                return full_response

        except requests.exceptions.ConnectionError:
            error_msg = (
                "Error: unable to connect to Ollama server. "
                "Make sure Ollama is running with 'ollama serve'."
            )
            print(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            print(error_msg)
            return error_msg
