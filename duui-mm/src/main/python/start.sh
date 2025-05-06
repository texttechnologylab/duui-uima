#!/bin/bash
set -e

echo "installing requirements..."
python -m pip install vllm

echo "Launching vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model 'microsoft/Phi-4-multimodal-instruct' \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend mp \
    --dtype auto \
    --trust-remote-code \
    --max-model-len 131072 \
    --enable-lora \
    --max-lora-rank 320 \
    --lora-extra-vocab-size 256 \
    --limit-mm-per-prompt audio=3,image=3 \
    --max-loras 2


VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
until curl -s http://localhost:8000/v1/models > /dev/null; do
  echo "Still waiting for vLLM..."
  sleep 2
done

echo "Launching FastAPI server..."
uvicorn duui-mm:app --host 0.0.0.0 --port 9714 --workers 1

wait $VLLM_PID
