import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_DIR = "/tmp/crisper_whisper"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print("Torch dtype:", torch_dtype)

model_id = "nyrahealth/CrisperWhisper"
print("Model:", model_id)

model_revision = "7aefea4c6c009ea7c47e6ab79247dfaf73d4c518"
print("Model revision:", model_revision)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    attn_implementation="eager",
    cache_dir=MODEL_DIR,
    revision=model_revision,
)
model.to(device)

processor = AutoProcessor.from_pretrained(
    model_id,
    cache_dir=MODEL_DIR,
    revision=model_revision,
)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    return_timestamps="word",
    chunk_length_s=30,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)
