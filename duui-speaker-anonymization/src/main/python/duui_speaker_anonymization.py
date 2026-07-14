from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import warnings
from functools import lru_cache
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Optional

import torch
import uvicorn
from cassis import load_typesystem
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUPPORTED_LANGUAGES = {"en", "de", "fr", "it", "es", "pt", "nl", "pl", "ru"}
DEFAULT_LANGUAGE = "en"

# ---------------------------------------------------------------------------
# Import anonymization modules from the repo
# ---------------------------------------------------------------------------
REPO_ROOT = os.environ.get("ANON_REPO_ROOT", os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

from anonymization.modules.text.recognition.whisper import WhisperASR
from anonymization.modules.speaker_embeddings.extraction.embedding_methods.style_embeddings import StyleEmbeddings
from anonymization.modules.speaker_embeddings.extraction.ims_speaker_extraction_methods import normalize_wave
from anonymization.modules.speaker_embeddings.speaker_embeddings import SpeakerEmbeddings
from anonymization.modules.speaker_embeddings.anonymization.gan_anon import GANAnonymizer
from anonymization.modules.prosody.extraction.ims_prosody_extraction import ImsProsodyExtractor
from anonymization.modules.tts.ims_tts import ImsTTS

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class DUUIRequest(BaseModel):
    audio: str
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def coerce_options(cls, v: Any) -> dict:
        if v is None or isinstance(v, list):
            return {}
        if not isinstance(v, dict):
            return {}
        return v

    @field_validator("audio", mode="before")
    @classmethod
    def coerce_audio(cls, v: Any) -> str:
        return "" if v is None else str(v)


class DUUIResponse(BaseModel):
    original_text: str
    anonymized_audio: str
    warning: Optional[str] = None


class DUUIDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: str


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    duui_tool_name: str = "DUUI Speaker Anonymization"
    duui_tool_version: str = "1.0"

    class Config:
        env_prefix = "duui_speaker_anonymization_"


settings = Settings()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="DUUI Speaker Anonymization",
    description="Speaker voice anonymization for TTLab DUUI",
    version="1.0",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Tim",
        "url": "https://www.texttechnologylab.org",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    logger.error("Bad request: %s", exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = await request.body()
    logger.error("422 validation errors: %s", exc.errors())
    logger.error("Raw body: %s", body.decode("utf-8", errors="replace"))
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": body.decode("utf-8", errors="replace")}),
    )


# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------

_this_dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(_this_dir, "communication.lua"), "rb") as _f:
    _communication_lua: str = _f.read().decode("utf-8")

with open(os.path.join(_this_dir, "typesystem.xml"), "rb") as _f:
    _typesystem = load_typesystem(_f)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    return JSONResponse(content=jsonable_encoder({
        "inputs": [],
        "outputs": [
            "org.texttechnologylab.annotation.type.AudioToken",
        ],
    }))


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    xml_content = _typesystem.to_xml().encode("utf-8")
    return Response(content=xml_content, media_type="application/xml")


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return _communication_lua


@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:
    return DUUIDocumentation(
        annotator_name=settings.duui_tool_name,
        version=settings.duui_tool_version,
        implementation_lang="Python",
    )


@app.post("/v1/process")
async def post_process(raw_request: Request) -> DUUIResponse:
    body = await raw_request.body()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RequestValidationError([
            {"type": "json_invalid", "loc": ("body",), "msg": str(exc), "input": body}
        ])
    request = DUUIRequest.model_validate(data)
    return _process(request)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", DEVICE)
if DEVICE.type == "cuda":
    logger.info("GPU: %s (memory: %.1f GB)", torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / 1e9)

models_dir = os.environ.get("MODELS_DIR", os.path.join(REPO_ROOT, "models"))

model_lock = Lock()


@lru_cache(maxsize=1)
def _load_anonymization_pipeline(language: str):
    """Load all models needed for anonymization."""
    logger.info("Loading anonymization models for language=%s onto %s", language, DEVICE)

    whisper_model = WhisperASR(
        model_path=os.path.join(models_dir, "whisper-large-v3"),
        device=DEVICE,
        lang=language,
    )

    embed_extractor = StyleEmbeddings(
        model_path=os.path.join(models_dir, "embedding_function.pt"),
        device=DEVICE,
    )

    gan_anon = GANAnonymizer(
        vec_type="style-embed",
        device=DEVICE,
        vectors_file=os.path.join(models_dir, "embedding_gan_generated_vectors.pt"),
        gan_model_path=os.path.join(models_dir, "embedding_gan.pt"),
        num_sampled=5000,
        sim_threshold=0.7,
        save_intermediate=True,
    )

    prosody_extractor = ImsProsodyExtractor(
        aligner_path=os.path.join(models_dir, "aligner.pt"),
        device=DEVICE,
        language=language,
    )

    tts = ImsTTS(
        hifigan_path=os.path.join(models_dir, "Avocodo.pt"),
        fastspeech_path=os.path.join(models_dir, "ToucanTTS_Meta.pt"),
        embedding_path=os.path.join(models_dir, "embedding_function.pt"),
        device=DEVICE,
        lang=language,
    )

    logger.info("All %d pipeline models loaded successfully", 5)
    if DEVICE.type == "cuda":
        logger.info("GPU memory allocated: %.1f GB", torch.cuda.memory_allocated(0) / 1e9)
    return whisper_model, embed_extractor, gan_anon, prosody_extractor, tts


def _load_pipeline(language: str):
    with model_lock:
        return _load_anonymization_pipeline(language)


# ---------------------------------------------------------------------------
# Business logic
# ---------------------------------------------------------------------------

def _decode_base64_audio(audio_b64: str) -> bytes:
    raw = base64.b64decode(audio_b64)
    if raw[:4] == b"RIFF":
        return raw
    wav_buf = io.BytesIO()
    audio_seg = AudioSegment.from_file(io.BytesIO(raw))
    audio_seg.export(wav_buf, format="wav")
    return wav_buf.getvalue()


def _process(request: DUUIRequest) -> DUUIResponse:
    options = request.options
    language = str(options.get("language", DEFAULT_LANGUAGE))
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{language}'. Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    if not request.audio:
        return DUUIResponse(original_text="", anonymized_audio="")

    logger.info("Anonymizing audio: language=%s", language)

    wav_bytes = _decode_base64_audio(request.audio)

    with NamedTemporaryFile(suffix=".wav") as audio_file:
        audio_file.write(wav_bytes)
        audio_file.flush()
        audio_path = audio_file.name

        whisper_model, embed_extractor, gan_anon, prosody_extractor, tts = \
            _load_pipeline(language)

        # Step 1: ASR
        logger.info("Step 1/4: Speech recognition (Whisper)")
        text = whisper_model.recognize_speech_of_audio(audio_path)
        logger.info("Transcript: %s", text.strip("~# "))

        # Step 2: Speaker embedding
        logger.info("Step 2/4: Speaker embedding extraction")
        import torchaudio
        signal, fs = torchaudio.load(audio_path)
        if signal.shape[0] > 1:
            signal = signal.mean(0, keepdim=True)
        norm_wave = normalize_wave(signal, fs, device=DEVICE)
        spk_vector = embed_extractor.extract_vector(audio=norm_wave, sr=fs)

        spk_embs = SpeakerEmbeddings(vec_type="style-embed", emb_level="utt",
                                     device=DEVICE)
        spk_embs.set_vectors(
            identifiers=["utt_0"],
            vectors=torch.stack([spk_vector]),
            speakers=["speaker1"],
            genders=["m"],
        )

        # Step 3: GAN anonymization
        logger.info("Step 3/4: GAN speaker anonymization")
        anon_embs = gan_anon.anonymize_embeddings(spk_embs, emb_level="utt")
        anon_vector = anon_embs.vectors[0]

        # Step 4: Prosody + TTS
        logger.info("Step 4/4: Prosody extraction + TTS synthesis")
        duration, pitch, energy, start_silence, end_silence = prosody_extractor.extract_prosody(
            transcript=text,
            ref_audio_path=audio_path,
            input_is_phones=False,
        )

        wav = tts.read_text(
            text=text,
            speaker_embedding=anon_vector,
            text_is_phones=False,
            duration=duration,
            pitch=pitch,
            energy=energy,
            start_silence=start_silence,
            end_silence=end_silence,
        )

    # Convert synthesized audio to WAV bytes then base64
    logger.info("Convert synthesized audio to WAV bytes then base64")
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(file=buf, data=wav, samplerate=tts.output_sr, format="WAV")
    buf.seek(0)
    anon_b64 = base64.b64encode(buf.read()).decode("utf-8")

    clean_text = text.strip("~# ")
    logger.info("Anonymization complete. Output: %d bytes base64", len(anon_b64))

    return DUUIResponse(original_text=clean_text, anonymized_audio=anon_b64)


if __name__ == "__main__":
    uvicorn.run("duui_speaker_anonymization:app", host="0.0.0.0", port=9714, workers=1)
