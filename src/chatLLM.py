import requests
import json
import sys
from typing import List, Tuple

# Constants
URL = "http://10.75.235.144:11434/api/generate"
MODEL = "llama3.2:3b"


class LLMClient:
    def __init__(self, max_history: int = 5):
        """
        LLM client with conversation memory (last N question/answer pairs)
        """
        self.max_history = max_history
        self.history: List[Tuple[str, str]] = []    # (user_question, llm_answer)
        self.alert_object_list = set(["person"])    #TODO Delete person    # Stored as a set of class names (strings)


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

    def remove_alert_object(self, class_name: str) -> None:
        """Remove an object class from the alert list"""
        self.alert_object_list.discard(class_name)

    def get_alert_objects(self) -> list:
        """Return current alert object classes as a list"""
        return list(self.alert_object_list)

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
            f"USER QUESTION: {user_text}\n"
            "INSTRUCTION: Answer the question taking into account "
            "the detected objects, their positions, and the previous conversation."
        )

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": True
        }

        print("\nREQUEST", "=" * 25)
        print(prompt)
        print("\nRESPONSE", "=" * 25)

        full_response = ""

        try:
            with requests.post(URL, json=payload, stream=True, timeout=30) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)

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
