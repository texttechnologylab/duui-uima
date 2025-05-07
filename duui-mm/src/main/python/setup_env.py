import os
from huggingface_hub import snapshot_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="microsoft/Phi-4-multimodal-instruct")
args = parser.parse_args()

model_path = snapshot_download(repo_id=args.model_path, revision='0af439b3adb8c23fda473c4f86001dbf9a226021')
vision_lora_path = os.path.join(model_path, "vision-lora")
speech_lora_path = os.path.join(model_path, "speech-lora")

# Write them to a shell export file
with open("env.sh", "w") as f:
    f.write(f"export VISION_LORA_PATH='{vision_lora_path}'\n")
    f.write(f"export SPEECH_LORA_PATH='{speech_lora_path}'\n")
