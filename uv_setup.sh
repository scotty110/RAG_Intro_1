#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

ENV_NAME=".rag"
uv venv $ENV_NAME --python 3.10

source $ENV_NAME/bin/activate

echo "Installing packages from requirements.txt..."
uv pip install -r requirements.txt

echo "Setup complete. To activate the environment in your terminal, run: source $ENV_NAME/bin/activate"
