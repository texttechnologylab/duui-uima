import base64
import io
import logging
import re
from pathlib import Path
from platform import python_version
from scipy.io import wavfile
from sys import version as sys_version
from tempfile import TemporaryDirectory
from time import time
from typing import List, Optional

import torch
import torchaudio
import torchaudio.transforms as T
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
import numpy as np
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
from starlette.responses import JSONResponse

# see https://github.com/nyrahealth/CrisperWhisper/blob/b7389e7d2371b94412ede5be4d9385dbebc005f8/utils.py#L1
def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output


# see https://github.com/nyrahealth/CrisperWhisper/blob/39b3f4c84a5ef217435956baf69095e07d33973c/app.py#L85
def process_audio_bytes(audio_bytes: bytes) -> torch.Tensor:
    """Process audio bytes to the required format."""
    audio_stream = io.BytesIO(audio_bytes)
    sr, y = wavfile.read(audio_stream)
    y = y.astype(np.float32)
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std
    transform = T.Resample(sr, 16000)
    waveform = transform(torch.unsqueeze(torch.tensor(y_normalized / 8), 0))
    # torchaudio.save("sample.wav", waveform, sample_rate=16000)
    return waveform


# TODO make configurable?
sns_min_silence_len = 5000  # Time in ms
sns_silence_thresh = -50    # Threshold of detected silence in db
sns_keep_silence = 2000     # Amount of silence kept at beginning and end for clearer audio
sns_seek_step = 100         # Step size for iterating over the segment in ms
sns_filter_filler = False   # Set to True to activate filter

SUPPORTED_LANGUAGES = [
    "EN",
    "DE",
]

crisper_whisper_transformers_version = "unknown"
with open("requirements.txt", "r", encoding="UTF-8") as fp:
    transformers_line = "transformers @ git+https://github.com/nyrahealth/transformers.git@"
    for line in fp:
        if line.startswith(transformers_line):
            crisper_whisper_transformers_version = line.split(transformers_line)[1].strip()
            break


class AudioToken(BaseModel):
    begin: int
    end: int
    timeStart: float
    timeEnd: float
    text: str


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class DUUIRequest(BaseModel):
    audio: str  # audio in base64


class DUUIResponse(BaseModel):
    audio_token: List[AudioToken]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]


class DUUICapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


class DUUIDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str]
    meta: Optional[dict]
    docker_container_id: Optional[str]
    parameters: Optional[dict]
    capability: DUUICapability
    implementation_specific: Optional[str]


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_crisper_whisper_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI CrisperWhisper")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

try:
    from src.main.python.crisper_whisper import pipe,model_revision as crisper_whisper_model_revision
except ModuleNotFoundError:
    from crisper_whisper import pipe, model_revision as crisper_whisper_model_revision

with open("src/main/lua/duui.lua", "rb") as f:
    communication = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(communication)

with open("src/main/resources/TypeSystem.xml", 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("UIMA typesystem:")
    logger.debug(typesystem_xml_content)


app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="CrisperWhisper DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Daniel Baumartz",
        "email": "baumartz@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": [],
        "outputs": ["org.texttechnologylab.annotation.type.AudioToken"]
    }
    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:
    capabilities = DUUICapability(
        supported_languages=sorted(SUPPORTED_LANGUAGES),
        reproducible=True  # really?
    )

    documentation = DUUIDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "crisper_whisper_transformers_version": crisper_whisper_transformers_version,
            "crisper_whisper_model_revision": crisper_whisper_model_revision,
        },
        docker_container_id="[TODO]",
        parameters={
            "language": sorted(SUPPORTED_LANGUAGES),
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


def process_audio(file_name: Path, export_location: Path):
    wav_audio = AudioSegment.from_file(file_name)

    chunk_times = detect_nonsilent(wav_audio, min_silence_len=sns_min_silence_len, silence_thresh=sns_silence_thresh, seek_step=sns_seek_step)
    chunks = split_on_silence(wav_audio, min_silence_len=sns_min_silence_len, silence_thresh=sns_silence_thresh, keep_silence=sns_keep_silence, seek_step=sns_seek_step)
    for i, chunk in enumerate(chunks):
        # Convert AudioSegment chunk to bytes
        chunk = chunk.set_channels(1)  # Convert to mono
        chunk_io = io.BytesIO()

        chunk.export(chunk_io, format="wav")
        audio_bytes = chunk_io.getvalue()

        # Process bytes into waveform
        waveform = process_audio_bytes(audio_bytes)

        # Save processed waveform using torchaudio
        chunk_name = f"current_{i}.wav"
        export_path = export_location / chunk_name
        print("Exporting processed:", chunk_name)
        torchaudio.save(str(export_path), waveform, sample_rate=16000)
    return chunk_times


@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    modification_timestamp_seconds = int(time())

    results = []
    meta = None
    modification_meta = None

    with TemporaryDirectory() as audio_dir:
        audio_dir = Path(audio_dir)

        # if this fails we stop processing
        audio_file = audio_dir / "audio.wav"
        with open(audio_file, "wb") as fp:
            fp.write(base64.b64decode(request.audio))

        chunks_dir = audio_dir / "chunks"
        chunks_dir.mkdir()

        times = process_audio(audio_file, chunks_dir)
        file_list = sorted(chunks_dir.glob("*"), key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

        # text_list = []
        chunk_list = []
        for i, file in enumerate(file_list):
            print("Transcribing sample: ", file)
            audio, sample_rate = torchaudio.load(str(file))
            audio = audio.squeeze(0)
            audio_array = audio.numpy()
            try:
                hf_pipeline_output = pipe(audio_array)
                crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
                # crisper_text = crisper_whisper_result["text"]
                # text_list.append(crisper_text)
                for chunk in crisper_whisper_result["chunks"]:
                    time_offset = float(times[i][0] / 1000 - (sns_keep_silence/1000))
                    if chunk["timestamp"][1] is None:
                        chunk["timestamp"] = (float(chunk["timestamp"][0] + time_offset),
                                              float(chunk["timestamp"][0] + time_offset))
                    else:
                        chunk["timestamp"] = (float(chunk["timestamp"][0] + time_offset),
                                              float(chunk["timestamp"][1] + time_offset))
                chunk_list.extend(crisper_whisper_result["chunks"])
            except RuntimeError as error:
                print("Skipped file: ", file, " due to RuntimeError:", error)

        # text = "".join(text_list)
        # json_dict = {"text": text, "words": chunk_list}

        current_length = 0
        for word in chunk_list:
            begin, end = word["timestamp"]

            text = word["text"]
            text_length = len(text)
            text_begin = current_length
            text_end = text_begin + text_length

            print(text, begin, end, text_begin, text_end)

            results.append(AudioToken(
                timeStart=float(begin),
                timeEnd=float(end),
                text=text,
                begin=text_begin,
                end=text_end
            ))

            current_length = text_end + 1

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=f"CrisperWhisper",
            modelVersion=crisper_whisper_model_revision
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), CrisperWhisper ({crisper_whisper_model_revision})"
        )

    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)
    
    return DUUIResponse(
        audio_token=results,
        meta=meta,
        modification_meta=modification_meta
    )
