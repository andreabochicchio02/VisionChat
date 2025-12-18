import subprocess

def speak(text: str):
    # Call espeak with an English voice and speed
    subprocess.run(["espeak", "-v", "en-us", "-s", "140", text], check=False)

if __name__ == "__main__":
    # Test phrases
    speak("Hello, this is a test of TTS on Raspberry Pi.")
    speak("If you hear this, text to speech is working.")
