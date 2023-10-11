import logging
from platform import python_version
from sys import version as sys_version
from time import time
from functools import lru_cache
from threading import Lock
from typing import List, Optional

import stanza
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str
    model_cache_size: int

    class Config:
        env_prefix = 'duui_sentencizer_stanza_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Stanza")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

UIMA_TYPE_SENTENCE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = {
    UIMA_TYPE_SENTENCE
}

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = {
    ""  # Text
}

SUPPORTED_LANGS = {
    "en",
    "de",
}


class TextImagerRequest(BaseModel):
    text: str
    len: int
    lang: str


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class Sentence(BaseModel):
    begin: int
    end: int


class TextImagerResponse(BaseModel):
    sentences: List[Sentence]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]


class TextImagerCapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str]
    meta: Optional[dict]
    docker_container_id: Optional[str]
    parameters: Optional[dict]
    capability: TextImagerCapability
    implementation_specific: Optional[str]


class TextImagerInputOutput(BaseModel):
    inputs: List[str]
    outputs: List[str]


typesystem_filename = 'TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)

lua_communication_script_filename = "communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
model_load_lock = Lock()


@lru_cache_with_size
def load_cache_stanza_model(model_name):
    logger.info("Loading Stanza model \"%s\"...", model_name)
    pipe = stanza.Pipeline(model_name, processors='tokenize', download_method=None)
    logger.info("Finished loading Stanza model \"%s\"", model_name)
    return pipe


def load_stanza_model(model_name):
    model_load_lock.acquire()

    err = None
    try:
        logger.info("Getting Stanza model \"%s\"...", model_name)
        pipe = load_cache_stanza_model(model_name)
    except Exception as ex:
        pipe = None
        err = str(ex)
        logging.exception("Failed to load Stanza model: %s", ex)

    model_load_lock.release()

    return pipe, err


app = FastAPI(
    title=settings.annotator_name,
    description="Stanza implementation for TTLab TextImager DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team - Daniel Baumartz",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=sorted(list(SUPPORTED_LANGS)),
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "stanza_version": stanza.__version__,
        },
        docker_container_id="[TODO]",
        parameters={},
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(
        inputs=TEXTIMAGER_ANNOTATOR_INPUT_TYPES,
        outputs=TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES
    )


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    sentences = []
    meta = None
    modification_meta = None

    try:
        pipe, pipe_err = load_stanza_model(request.lang)
        if pipe is None:
            raise Exception(f"Stanza model \"{request.lang}\" could not be loaded: {pipe_err}")

        doc = pipe(request.text)
        for sent in doc.sentences:
            sentences.append(Sentence(
                begin=sent.tokens[0].start_char,
                end=sent.tokens[-1].end_char
            ))

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName="Stanza",
            modelVersion=stanza.__version__
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), Stanza ({stanza.__version__})"
        )

    except Exception as ex:
        logger.exception(ex)

    logger.debug(sentences)
    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    return TextImagerResponse(
        sentences=sentences,
        meta=meta,
        modification_meta=modification_meta,
    )
