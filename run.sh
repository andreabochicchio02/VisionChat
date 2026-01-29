#!/usr/bin/env bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source venv/bin/activate

# Move to project directory
cd VisionChat/VisionChat/src/

# Run the application
python3 app.py
