#!/bin/bash
set -e

#echo "installing VLLM..."
#python -m pip install vllm
# Load them
source ./env.sh

echo "SPEECH_LORA_PATH=$SPEECH_LORA_PATH"
echo "VISION_LORA_PATH=$VISION_LORA_PATH"


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
    --limit-mm-per-prompt audio=10,image=10 \
    --max-loras 2 \
    --lora-modules speech=$SPEECH_LORA_PATH vision=$VISION_LORA_PATH \
    &

VLLM_PID=$!

echo "Waiting for vLLM to be ready..."
until python -c "import requests; requests.get('http://localhost:8000/v1/models')" > /dev/null 2>&1; do
  echo "Still waiting for vLLM..."
  sleep 5
done


echo "Launching FastAPI server..."
uvicorn duui-mm:app --host 0.0.0.0 --port 9714 --workers 1

wait $VLLM_PID
