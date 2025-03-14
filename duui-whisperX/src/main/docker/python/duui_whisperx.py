import base64
import logging
from platform import python_version
from sys import version as sys_version
from tempfile import NamedTemporaryFile
from time import time
from typing import List, Optional

import torch
import whisperx
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse


# TODO keep up to date with requirements.txt
whisperx_version = "3.3.1"


# Token
class AudioToken(BaseModel):
    """
    org.texttechnologylab.annotation.type.AudioToken
    """
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


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # audio in base64
    audio: str
    language: str
    model: str = "large-v2"
    batch_size: int = 16


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of annotated:
    # - audiotoken
    audio_token: List[AudioToken]
    language: str


class DUUICapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


# Documentation response
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
        env_prefix = 'duui_whisperx_'


# settings + cache
settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI whisperX")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

# Start fastapi
app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="WhisperX audio transcription",
    description="Audio transcription for TTLab DUUI",
    version="2.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Daniel Bundan",
        "url": "bundan.me",
        "email": "s1486849@stud.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Load the Lua communication script
communication = "communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", communication)
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(communication)

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'typesystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": [],
        "outputs": ["org.texttechnologylab.annotation.type.AudioToken"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:
    capabilities = DUUICapability(
        supported_languages=SUPPORTED_LANGS,
        reproducible=True
    )

    documentation = DUUIDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "whisperx_version": whisperx_version
        },
        docker_container_id="[TODO]",
        parameters={},
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    modification_timestamp_seconds = int(time())

    results = []
    meta = None
    modification_meta = None

    language = None
    if request.language:
        language = request.language
    print("Language:", language)

    # TODO detect only once at startup?
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    compute_type = "float16" if torch.cuda.is_available() else "int8"

    # Load different pipeline depending on CUDA availability
    asr_options = {"word_timestamps": True}

    with NamedTemporaryFile() as audio_file:
        # if this fails we stop processing
        with open(audio_file.name, "wb") as fp:
            fp.write(base64.b64decode(request.audio))

        model = whisperx.load_model(
            request.model, device, compute_type=compute_type, language=language, asr_options=asr_options,
            local_files_only=True, download_root="/tmp/whisperx"
        )
        audio = whisperx.load_audio(audio_file.name)
        result = model.transcribe(audio, batch_size=request.batch_size)

        # use language detected by the model if not provided
        if not language:
            language = result["language"]
            print("Using detected language:", language)

        alignment_model, metadata = whisperx.load_align_model(language_code=language, device=device)
        aligned_result = whisperx.align(result["segments"], alignment_model, metadata, audio_file.name, device)

        current_length = 0
        for word in aligned_result["word_segments"]:
            audio_start = word.get("start")
            audio_end = word.get("end")
            text = word.get("word").strip()

            if audio_start is None or audio_end is None:  # If segment is not spoken out loud, such as '-'
                continue

            if len(text) == 0 and audio_start == audio_end:  # If segment contains no information
                continue

            results.append(AudioToken(
                timeStart=float(audio_start),
                timeEnd=float(audio_end),
                text=text,
                begin=current_length,
                end=current_length + len(text)
            ))

            if len(text) > 0:
                current_length += len(text) + 1

            meta = AnnotationMeta(
                name=settings.annotator_name,
                version=settings.annotator_version,
                modelName=f"whisperX {request.model}",
                modelVersion=whisperx_version
            )

            modification_meta = DocumentModification(
                user=settings.annotator_name,
                timestamp=modification_timestamp_seconds,
                comment=f"{settings.annotator_name} ({settings.annotator_version}), whisperX ({whisperx_version})"
            )

    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)
    
    return DUUIResponse(
        audio_token=results,
        language=language
    )


#if __name__ == "__main__":
#  uvicorn.run("duui_whisperx:app", host="0.0.0.0", port=9714, workers=1)
