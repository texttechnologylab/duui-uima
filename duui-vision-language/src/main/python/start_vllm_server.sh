#!/bin/bash
set -e

export VLLM_PLATFORM=cuda

pip uninstall libtpu-nightly jax jaxlib -y


echo "Launching vLLM server..."
# Load and run the model:
vllm serve "allenai/Molmo-7B-D-0924" \
 --trust-remote-code --device auto \
 --revision ac032b93b84a7f10c9578ec59f9f20ee9a8990a2 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --dtype auto \
  --trust-remote-code \
  &

VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
until curl -s http://localhost:8000/v1/models > /dev/null; do
  echo "Still waiting for vLLM..."
  sleep 5
done

echo "Launching FastAPI server..."
uvicorn duui-mm:app --host 0.0.0.0 --port 9714 --workers 1

wait $VLLM_PID
