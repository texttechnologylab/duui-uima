import logging
import math
import os
import sys
from functools import lru_cache
from typing import Final, Dict, List, Optional, Iterable, TypeVar

import flair
from fastapi import FastAPI, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse, JSONResponse
from flair.data import Sentence
from flair.models import SequenceTagger
from pydantic import BaseModel

T = TypeVar("T")

logger = logging.getLogger("fastapi")

MODEL_CACHE_SIZE: Final[int] = int(os.environ.get("MODEL_CACHE_SIZE", 1))
logger.info(f"MODEL_CACHE_SIZE={MODEL_CACHE_SIZE}")
BATCH_SIZE: Final[int] = int(os.environ.get("FLAIR_BATCH_SIZE", 128))
logger.info(f"BATCH_SIZE={BATCH_SIZE}")

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="Flair POS - DUUI",
    description="Flair POS Tagger for the TTLab TextImager DUUI",
    version=os.environ.get("FLAIR_POS_VERSION", "0.0.1"),
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "manuel.stoeckel@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

logger.debug("Loading Lua communication layer from file")
with open("communication_layer.lua", "r", encoding="utf-8") as f:
    lua_communication_script = f.read()

logger.debug("Loading type system from file")
with open("dkpro-core-types.xml", "r", encoding="utf-8") as f:
    type_system = f.read()

lang_code_to_model_map: Final[Dict[str, str]] = {
    "en": "pos",  # English
    "en-fast": "pos-fast",  # English Fast
    "en-upos": "upos",  # English UPOS
    "en-upos-fast": "upos-fast",  # English UPOS Fast
    "multi": "pos-multi",  # Multilingual
    "multi-fast": "pos-multi-fast",  # Multilingual Fast
    "ar": "ar-pos",  # Arabic
    "de": "de-pos",  # German
    "de-twitter": "de-pos-tweets",  # German Tweets
    "da": "da-pos",  # Danish
    "ms": "ml-pos",  # Malay
    "ms-upos": "ml-upos",  # Malay UPOS
    "pt": "pt-pos-clinical",  # Portuguese
    "uk": "pos-ukrainian",  # Ukrainian
}
supported_languages: Final[List[str]] = list(
    sorted(lang_code_to_model_map.keys()))


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=type_system, media_type="application/xml")


# Capabilities
class TextImagerCapability(BaseModel):
    # List of supported languages by the annotator
    # TODO how to handle language?
    # - ISO 639-1 (two letter codes) as default in meta data
    # - ISO 639-3 (three letters) optionally in extra meta to allow a finer mapping
    supported_languages: List[str]
    # Are results on same inputs reproducible without side effects?
    reproducible: bool


# Documentation response
class TextImagerDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str]
    # Optional map of additional meta data
    meta: Optional[dict]
    # Docker container id, if any
    docker_container_id: Optional[str]
    # Optional map of supported parameters
    parameters: Optional[dict]
    # Capabilities of this annotator
    capability: TextImagerCapability
    # Analysis engine XML, if available
    implementation_specific: Optional[str]


class DkproPos(BaseModel):
    """
    Models the DKPRO POS type, de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS
    """

    # Inherited from uima.tcas.Annotation
    begin: int
    # Inherited from uima.tcas.Annotation
    end: int
    # Fine-grained POS tag. This is the tag as produced by a POS tagger or obtained from a reader.
    pos_value: str
    # Coarse-grained POS tag. This may be produced by a POS tagger or reader in addition to the fine-grained tag.
    coarse_value: str


class DkproSentence(BaseModel):
    """
    Models the DKPRO Sentence type, de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence
    """

    offset: int
    coveredText: str


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=supported_languages, reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name="Flair POS - DUII",
        version="0.0.1",
        implementation_lang="Python",
        meta={"python_version": sys.version,
              "flair_version": flair.__version__},
        docker_container_id="docker.texttechnologylab.org/flair/pos:latest",
        parameters={
            "language": "de",
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


class Message(BaseModel):
    message: str


class TextImagerRequest(BaseModel):
    text: str
    language: str
    sentences: List[DkproSentence]


class TextImagerResponse(BaseModel):
    tags: List[DkproPos]


@lru_cache(maxsize=MODEL_CACHE_SIZE)
def load_model(lang: str) -> SequenceTagger:
    return SequenceTagger.load(lang)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=400)


def batcher(iterable: List[T], batch_size=BATCH_SIZE) -> Iterable[List[T]]:
    _len = len(iterable)
    for start in range(0, _len, batch_size):
        yield list(iterable[start:start + batch_size])


def flatten(iterable: Iterable[Iterable[T]]) -> Iterable[T]:
    for it in iterable:
        yield from it


@app.post(
    "/v1/process",
    response_model=TextImagerResponse,
    responses={
        400: {
            "model": Message,
            "description": "There was an error with the request",
        },
    },
)
def post_process(request: TextImagerRequest):
    language = request.language
    if language not in supported_languages:
        supported_lang_string = ", ".join(supported_languages)
        return JSONResponse(
            status_code=400,
            content={
                "message": f"The selected language '{language}' is not supported. "
                           f"Supported languages: {supported_lang_string}"
            },
        )
    model = load_model(lang_code_to_model_map[language])
    if request.sentences:
        total_batches = math.ceil(len(request.sentences) / BATCH_SIZE * 1.)

        def process_verbose(idx, batch: List[T]) -> Iterable[T]:
            logger.info(f"Processing batch {idx}/{total_batches}")
            return process_batch(model, request.text, batch)

        pos_tags = list(flatten(
            process_verbose(idx, batch)
            for idx, batch in enumerate(batcher(request.sentences, BATCH_SIZE), start=1)
        ))
        return TextImagerResponse(tags=pos_tags)
    else:
        return JSONResponse(
            status_code=400,
            content={
                "message": "The input MUST be sentence segmented for the Flair POS tagger to work."
            },
        )


def process_batch(model: SequenceTagger, text: str, batch: List[DkproSentence]) -> Iterable[DkproPos]:
    text_len = len(text)
    sentences: List[Sentence] = []
    for dkpro_sentence in batch:
        sentences.append(
            Sentence(
                dkpro_sentence.coveredText,
                start_position=dkpro_sentence.offset,
            )
        )
    model.predict(sentences)
    for sentence in sentences:
        for label in sentence.get_labels():
            begin = label.data_point.start_position + sentence.start_position
            end = label.data_point.end_position + sentence.start_position
            value = label.value
            yield DkproPos(begin=begin, end=end,
                           pos_value=value, coarse_value="")
