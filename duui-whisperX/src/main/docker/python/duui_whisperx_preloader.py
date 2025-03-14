import torch
import whisperx


MODELS_LIST = [
    "large-v2",
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

for model_name in MODELS_LIST:
    whisperx.load_model(model_name, device, download_root="/tmp/whisperx")
    print(f"Model {model_name} loaded successfully.")
