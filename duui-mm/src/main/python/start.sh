#!/bin/bash
set -e

source ./env.sh

export VLLM_SERVER_DEV_MODE=1

# Start Qwen2.5 VL-7B-Instruct using vLLM on a different port
echo "Launching Qwen2.5-VL-7B-Instruct..."
python -m vllm.entrypoints.openai.api_server \
    --model 'Qwen/Qwen2.5-VL-7B-Instruct' \
    --revision "cc594898137f460bfe9f0759e9844b3ce807cfb5" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend mp \
    --dtype auto \
    --trust-remote-code \
    --enable-sleep-mode \
    --port 6659 &
QWEN_PID=$!

# Wait for Qwen to be ready
until python -c "import requests; requests.get('http://localhost:6659/v1/models')" > /dev/null 2>&1; do
  echo "Waiting for Qwen on port 6659..."
  sleep 5
done

# Sleep Qwen initially using Python requests
python -c "import requests; requests.post('http://localhost:6659/sleep')" || true

# Start Microsoft Phi-4 using vLLM
echo "Launching vLLM server for Phi-4..."
python -m vllm.entrypoints.openai.api_server \
    --model 'microsoft/Phi-4-multimodal-instruct' \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --distributed-executor-backend mp \
    --dtype auto \
    --enable-sleep-mode\
    --trust-remote-code \
    --max-model-len 131072 \
    --enable-lora \
    --max-lora-rank 320 \
    --lora-extra-vocab-size 256 \
    --limit-mm-per-prompt audio=10,image=10 \
    --max-loras 2 \
    --lora-modules speech=$SPEECH_LORA_PATH vision=$VISION_LORA_PATH \
    --port 6658 &
VLLM_PID=$!

# Wait for vLLM to be ready
until python -c "import requests; requests.get('http://localhost:6658/v1/models')" > /dev/null 2>&1; do
  echo "Waiting for vLLM on port 6658..."
  sleep 5
done


# Start DUUI FastAPI app
echo "Launching FastAPI server..."
uvicorn duui-mm:app --host 0.0.0.0 --port 9714 --workers 1

# Wait for background processes
wait $VLLM_PID
wait $QWEN_PID
