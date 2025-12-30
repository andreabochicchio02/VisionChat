### LLM

- **llama3.1:3b** (the one we use)
- Larger models (**llama3.1:8b**) do not significantly improve the quality of the responses, but they increase latency.
- Reasoning models (**deepseek-r1:1.5b**) do not work well because they are too small; larger reasoning models introduce excessive latency.  


<br>
<br>


### TTS

- **vosk-model-small-it-0.22** works very well, but it is in Italian (see `testMIC/Vosk-it.py`).
- **vosk-model-en-0.22** is the best model, but its size is 1.8 GB, so we cannot use it.
- **vosk-model-en-us-0.22-lgraph** makes fewer errors than the previous one, but has latency on the order of seconds (see `testMIC/Vosk-en.py`).
- **vosk-model-small-en-us-0.15** (the one we use) makes a few errors, but the latency is extremely low.
-- **OpenAI's Whisper‑Tiny** has accuracy similar to Vosk‑en‑us‑0.15, but it has slightly higher latency and requires more resources (see `testMIC/Whisper-Tiny.py`).
