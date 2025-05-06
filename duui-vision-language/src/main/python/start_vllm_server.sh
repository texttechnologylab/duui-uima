#!/bin/bash
set -e

# Install vLLM from pip:
pip install vllm

echo "Launching vLLM server..."
# Load and run the model:
vllm serve "allenai/Molmo-72B-0924"

VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
until curl -s http://localhost:8000/v1/models > /dev/null; do
  echo "Still waiting for vLLM..."
  sleep 2
done

echo "Launching FastAPI server..."
uvicorn duui-mm:app --host 0.0.0.0 --port 9714 --workers 1

wait $VLLM_PID
