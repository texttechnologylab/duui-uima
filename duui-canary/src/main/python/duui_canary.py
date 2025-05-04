import base64
import json
import logging
import subprocess
import sys
from platform import python_version
from sys import version as sys_version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import List, Optional

import librosa
import soundfile as sf
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from nemo import __version__ as nemo_version
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# TODO
SUPPORTED_LANGUAGES = [
    "en",
    "de"
]

# TODO
SUPPORTED_MODELS = [

]


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
    audio: str  # base64 encoded
    language: str
    # model: str = "large-v2"
    # batch_size: int = 16


class DUUIResponse(BaseModel):
    audio_token: List[AudioToken]
    meta: AnnotationMeta
    modification_meta: DocumentModification


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
        env_prefix = 'duui_canary_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI NeMo Canary")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

with open("src/main/lua/communication.lua", 'rb') as fp:
    communication = fp.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(communication)

with open("src/main/resources/typesystem.xml", 'rb') as fp:
    typesystem = load_typesystem(fp)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("UIMA Typesystem:")
    logger.debug(typesystem_xml_content)

app = FastAPI(
    title=settings.annotator_name,
    description="DUUI Canary",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "email": "baumartz@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    return {
        "inputs": [
        ],
        "outputs": [
            "org.texttechnologylab.annotation.type.AudioToken"
        ]
    }

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
        reproducible=False
    )

    documentation = DUUIDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "nemo_version": nemo_version
        },
        docker_container_id="[TODO]",
        parameters={
            "language": sorted(SUPPORTED_LANGUAGES),
            # "model": sorted(SUPPORTED_MODELS),
            # "batch_size": "int",
            # "allow_download": "bool",
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation

@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    modification_timestamp_seconds = int(time())

    results = []
    meta = None
    modification_meta = None

    language = None
    if request.language:
        language = request.language
    logger.info("Language: %s", language)

    # TODO
    model_name = "nvidia/canary-1b-flash"

    # TODO delete by default
    with TemporaryDirectory(delete=False) as temp_dir:
        temp_path = Path(temp_dir)

        # TODO convert to wav or task of user?
        # audio_temp = temp_path / "audio.unk"
        audio_temp = temp_path / "audio.wav"
        with open(audio_temp, "wb") as fp:
            fp.write(base64.b64decode(request.audio))

        # convert audio for nemo/canary
        audio, sample_rate = librosa.load(audio_temp, sr=16000, mono=True)

        audio_path = temp_path / "audio.wav"
        sf.write(audio_temp, audio, sample_rate, format="WAV")

        manifest = {
            "audio_filepath": str(audio_path),
            "taskname": "asr",
            "source_lang": language,
            "target_lang": language,
            "pnc": "yes"
        }
        manifest_path = temp_path / "manifest.json"
        with open(manifest_path, "w", encoding="UTF-8") as fp:
            json.dump(manifest, fp)

        output_path = temp_path / "output.json"
        command = [
            f"{sys.executable}",
            "./speech_to_text_aed_chunked_infer.py",
            f"pretrained_name={model_name}",
            f"dataset_manifest={manifest_path}",
            f"output_filename={output_path}",
            "chunk_len_in_secs=10",
            "timestamps=True",
        ]
        process = subprocess.run(command, capture_output=True, text=True)

        # TODO
        if process.returncode == 0:
            print("Command succeeded:")
            print(process.stdout)
        else:
            print("Command failed with return code", process.returncode)
            print("Error output:", process.stderr)

        with open(output_path, "r", encoding="UTF-8") as fp:
            results = json.load(fp)

        current_length = 0
        full_text = ""
        for word in results["word"]:
            # TODO offsets starten nicht bei 0?
            """
            {
              "word": "Bricht",
              "start_offset": 91,
              "end_offset": 97,
              "start": 7.28,
              "end": 7.76
            },
            """
            audio_start = word.get("start")
            audio_end = word.get("end")
            text = word.get("word").strip()

            if audio_start is None or audio_end is None:
                continue

            if len(text) == 0 and audio_start == audio_end:
                continue

            results.append(AudioToken(
                timeStart=float(audio_start),
                timeEnd=float(audio_end),
                text=text,
                begin=current_length,
                end=current_length + len(text)
            ))

            if len(text) > 0:
                full_text += text + " "
                current_length += len(text) + 1

            # TODO segments
            """
            {
              "segment": "Bricht euch den Hals und wir lachen uns .",
              "start_offset": 91,
              "end_offset": 123,
              "start": 7.28,
              "end": 9.84
            },
            """

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=f"NeMo {model_name}",
            modelVersion=nemo_version
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), NeMo {model_name} ({nemo_version})"
        )

    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)
    
    return DUUIResponse(
        audio_token=results,
        # language=language,
        meta=meta,
        modification_meta=modification_meta
    )
