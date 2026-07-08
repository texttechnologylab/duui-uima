import logging
from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional, Any

from cassis import load_typesystem
from fastapi import FastAPI, Response, Depends, Body, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str
    model_cache_size: int = 1

    class Config:
        env_prefix = 'duui_sentence_transformers_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Sentence Transformers")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    "org.texttechnologylab.uima.type.Embedding"
]

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = [
    "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
]

SUPPORTED_LANGS = [
    # all
]


class Sentence(BaseModel):
    begin: int
    end: int
    text: str


class TextImagerRequest(BaseModel):
    sentences: List[Sentence]
    model_name: str
    batch_size: int = 32


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class Embedding(BaseModel):
    begin: int
    end: int
    vector: List[float]


class TextImagerResponse(BaseModel):
    embeddings: Optional[List[Embedding]]
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


typesystem_filename = 'src/main/resources/TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)

lua_communication_script_filename = "src/main/lua/communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

app = FastAPI(
    title=settings.annotator_name,
    description="TTLab TextImager DUUI Sentence Transformers",
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
        supported_languages=SUPPORTED_LANGS,
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
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


lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
model_load_lock = Lock()


@lru_cache_with_size
def load_cache_model(model_name):
    sha = HfApi().model_info(model_name, revision="main").sha
    model = SentenceTransformer(model_name, revision=sha)
    return model, sha


def load_model(model_name):
    model_load_lock.acquire()

    try:
        model, sha = load_cache_model(model_name)
    except Exception as ex:
        model = None
        sha = None
        logging.exception("Failed to load model: %s", ex)
    finally:
        model_load_lock.release()

    return model, sha


# Compatibility shim:
# DUUI send a JSON object as a JSON-encoded string.
# Pydantic/FastAPI newer versions validate the parsed body strictly as a str,
# so we explicitly parse that string with model_validate_json().
def get_text_imager_request(body: Any = Body(...)) -> TextImagerRequest:
    try:
        if isinstance(body, bytes):
            # raw JSON bytes
            return TextImagerRequest.model_validate_json(body)

        if isinstance(body, str):
            # JSON string containing the object
            return TextImagerRequest.model_validate_json(body)

        # normal FastAPI JSON parsing case: dict
        return TextImagerRequest.model_validate(body)

    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=exc.errors(include_input=False),
        )


@app.post("/v1/process")
def post_process(request: TextImagerRequest = Depends(get_text_imager_request)) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    embeddings = None
    meta = None
    modification_meta = None

    try:
        model, model_version = load_model(request.model_name)

        sentences = [sentence.text for sentence in request.sentences]
        vectors = model.encode(sentences, batch_size=request.batch_size)

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=request.model_name,
            modelVersion=model_version
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), {request.model_name} ({model_version})"
        )

        if len(vectors) != len(sentences):
            logging.error("embeddings does not match sentences")

        embeddings = []
        for sentence, vector in zip(request.sentences, vectors):
            embeddings.append(Embedding(
                begin=sentence.begin,
                end=sentence.end,
                vector=vector,
            ))

    except Exception as ex:
        logger.exception(ex)

    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    response = TextImagerResponse(
        embeddings=embeddings,
        meta=meta,
        modification_meta=modification_meta,
    )
    # print(response)
    return response
