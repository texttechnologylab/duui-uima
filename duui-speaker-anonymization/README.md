# DUUI Speaker Anonymization

DUUI component for multilingual speaker voice anonymization, based on the [IMS speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization/) system (Interspeech 2024).

## System Overview

The pipeline anonymizes speech in three steps:

1. **ASR** — Transcribes input audio using OpenAI Whisper large-v3
2. **Speaker embedding anonymization** — Extracts speaker embeddings and replaces them with GAN-generated artificial embeddings
3. **TTS synthesis** — Re-synthesizes speech using the anonymized embedding with [IMS Toucan](https://github.com/DigitalPhonetics/IMS-Toucan)

## Supported Languages

English (en), German (de), French (fr), Italian (it), Spanish (es), Portuguese (pt), Dutch (nl), Polish (pl), Russian (ru)

## Model Downloads

Download the following models into `src/main/python/models/`:

| File | Size | Purpose | Download |
|------|------|---------|----------|
| `embedding_function.pt` | ~8 MB | Speaker embedding encoder | [IMS-Toucan v2.5](https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_function.pt) |
| `embedding_gan.pt` | ~0.7 MB | GAN speaker embedding generator | [IMS-Toucan v2.5](https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_gan.pt) |
| `aligner.pt` | ~211 MB | Prosody aligner | [IMS-Toucan v2.5](https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/aligner.pt) |
| `ToucanTTS_Meta.pt` | ~177 MB | Multilingual TTS model | [IMS-Toucan v2.5](https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/ToucanTTS_Meta.pt) |
| `Avocodo.pt` | ~53 MB | Vocoder | [IMS-Toucan v2.5](https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/Avocodo.pt) |

```bash
# Example: download all models
cd src/main/python/models/
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_function.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/embedding_gan.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/aligner.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/ToucanTTS_Meta.pt
wget https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5/Avocodo.pt
```

The Whisper model (`whisper-large-v3`) is downloaded automatically on first use and cached in `src/main/python/models/whisper-large-v3/`.

## Running

### Start the server

```bash
cd src/main/python/
python duui_speaker_anonymization.py
```

The server starts on `http://0.0.0.0:9714`.

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/communication_layer` | Returns Lua communication layer |
| `GET` | `/v1/typesystem` | Returns UIMA type system XML |
| `POST` | `/v1/process` | Anonymizes audio |

### Process request

JSON body:

```json
{
  "audio": "<base64-encoded WAV audio>",
  "options": {
    "language": "en"
  }
}
```

Response:

```json
{
  "original_text": "transcribed text",
  "anonymized_audio": "<base64-encoded anonymized WAV>"
}
```

## Credits

This component wraps the speaker anonymization system developed at the Institute for Natural Language Processing (IMS), University of Stuttgart:

- Original repository: [github.com/DigitalPhonetics/speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization/)
- TTS toolkit: [IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan)
- Papers: [Meyer et al., Interspeech 2022](https://doi.org/10.21437/Interspeech.2022-10703) · [Meyer et al., SLT 2022](https://doi.org/10.1109/SLT54892.2023.10022601) · [Meyer et al., ICASSP 2023](https://doi.org/10.1109/ICASSP49357.2023.10096607) · [Meyer et al., Interspeech 2024](https://doi.org/10.21437/Interspeech.2024-1615)
