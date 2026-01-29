# Vision Chat: Real-Time Vision and Voice Assistant

**Vision Chat** is an interactive, intelligent system capable of performing real-time object detection and natural language interaction on embedded hardware. Built on a **Raspberry Pi 3 Model B+**, the system allows users to ask questions about the live camera view ("What do you see?") or set proactive alerts ("Notify me when a bottle appears") using natural speech.

This project validates a hybrid architecture where perception tasks (vision/audio) run locally on resource-constrained hardware, while complex reasoning is offloaded to a local Large Language Model (LLM).

## Table of Contents
- [System Overview](#system-overview)
- [Key Features](#key-features)
- [Hardware Requirements](#hardware-requirements)
- [Software & OS Requirements](#software--os-requirements)
- [Installation Guide](#installation-guide)
- [Usage](#usage)
## System Overview
The system combines Computer Vision and Natural Language Processing to create an assistive device. It captures video frames, detects objects using **SSD MobileNet V3**, and listens for user voice commands. It supports:
* **Reactive Interaction:** Answering questions about the scene.
* **Proactive Interaction:** Monitoring the video feed to alert the user when specific objects or motion are detected based on user-defined rules.

Computation is distributed:
1.  **Raspberry Pi:** Handles Object Detection, Motion Detection, Speech-to-Text (STT), Text-to-Speech (TTS), and the Web Interface.
2.  **External Host PC:** Runs the LLM (Llama 3.2) via Ollama to handle logic and natural language generation.

## Key Features
* **Real-Time Object Detection:** Uses `SSD MobileNet V3 Large` (COCO dataset) optimized for the Raspberry Pi.
* **Voice Interface:**
    * **STT:** Offline speech recognition using **Vosk**.
    * **TTS:** Lightweight speech synthesis using **Pico2wave**.
* **Smart Notifications:** Users can verbally subscribe to events (e.g., "Alert me if a person appears").
* **Motion Detection:** Lightweight algorithm to detect scene changes.
* **Dual Language Support:** Configurable for both **English** and **Italian**.
* **Web Dashboard:** A Flask-based UI to view the live video stream and conversation history.

## Hardware Requirements
To replicate this setup, you will need:

* **Embedded Device:** Raspberry Pi 3 Model B+.
* **Vision:** Raspberry Pi Camera Module v2 connected via CSI.
* **Audio Input:** USB Microphone.
* **Audio Output:** Speakers or Headphones.
* **LLM Server:** A separate PC/Laptop (connected to the same network) to run Ollama.

## Software & OS Requirements

* **OS:** Raspberry Pi OS (Legacy, 64-bit) Lite.
  * **Version:** A port of Debian Bookworm with security updates and No Desktop Environment (Headless/Lite version) to maximize resources for the AI models.
* **External Dependencies (Host PC):** 
  * **Ollama:** Must be installed on the host PC to serve the Llama 3.2 model.

## Installation Guide

### 1. System Preparation (Raspberry Pi)
Ensure your Raspberry Pi is running the specific OS mentioned above. Update the system and install system-level dependencies for audio and image processing:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv portaudio19-dev libatlas-base-dev libttspico-utils ffmpeg libsm6 libxext6

```

### 2. Clone the Repository

Clone this repository to your Raspberry Pi:

```bash
git clone https://github.com/andreabochicchio02/VisionChat
cd VisionChat
```

### 3. Set up the Python Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Python Dependencies

Install the required Python libraries using `pip`.

```bash
pip install -r requirements.txt

```

### 5. Setup the LLM (External PC)

1. Install [Ollama](https://ollama.com/) on your external PC.
2. Pull the Llama 3.2 model:
```bash
ollama pull llama3.2:3b
```


3. Ensure the Ollama server is running and accessible via the network (you may need to configure `OLLAMA_HOST` environment variables on the PC to accept external connections).

## Usage

### 1. Hardware Connection

Connect the peripherals to the Raspberry Pi in the following order:

1. Camera Module (ensure the ribbon cable is seated correctly).
2. USB Microphone.
3. Speakers/Headphones.
4. Power on the Raspberry Pi.

### 2. Configuration

Ensure the Raspberry Pi and the Host PC are on the **same Wi-Fi/LAN network**.

### 3. Run the Application

Navigate to the source directory and execute the main application:

```bash
cd src
python3 app.py
```
