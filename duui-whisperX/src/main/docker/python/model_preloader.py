import torch
import whisperx


MODEL_DIR = "/tmp/whisperx"

SUPPORTED_MODELS = [
    "large-v2",
    "large-v3",
]

SUPPORTED_LANGUAGES = [
    "en",
    "de",
    "ru",
]

# TODO only cpu detected on docker build
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

compute_type = "float16" if torch.cuda.is_available() else "int8"

asr_options = {"word_timestamps": True}

for model_name in SUPPORTED_MODELS:
    for language in SUPPORTED_LANGUAGES:
        whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=language,
            asr_options=asr_options,
            download_root=MODEL_DIR
        )
        whisperx.load_align_model(
            language_code=language,
            device=device,
            model_dir=MODEL_DIR
        )
        print(f"Model {model_name} with language {language} loaded successfully.")
