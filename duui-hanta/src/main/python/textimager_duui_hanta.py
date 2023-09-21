import logging
import os

from cassis import load_typesystem
from datetime import datetime
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from functools import lru_cache
from HanTa import HanoverTagger as ht
from platform import python_version
from pydantic import BaseSettings, BaseModel
from sys import version as sys_version
from threading import Lock
from time import time
from typing import List, Optional

# Settings
# These are automatically loaded from env variables
class Settings(BaseSettings):
    # Name of this annotator
    textimager_hanta_annotator_name: str
    # Version of this annotator
    textimager_hanta_annotator_version: str
    # Log level
    textimager_hanta_log_level: str
    # Model name
    textimager_hanta_model_name: str


# Load settings from env vars
settings = Settings()

# Init logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=settings.textimager_hanta_log_level,#logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI HanTaLemmatizer")
logger.info("Name: %s", settings.textimager_hanta_annotator_name)
logger.info("Version: %s", settings.textimager_hanta_annotator_version)

# Type names needed by this annotator
UIMA_TYPE_LEMMA = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma"
# Note: Without any extra meta types or subtypes
TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    UIMA_TYPE_LEMMA,
]

class TextImagerRequest(BaseModel):
    # The text to process
    text: str
    # The texts language
    lang: str
    #tokens: List
    #
    sents: Optional[list]
    tokens: List
    #tokens: Optional[list]
    #
    # Optional map/dict of parameters
    # TODO how with lua?
    parameters: Optional[dict]

# UIMA type: adds metadata to each annotation
class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str
    spacyVersion: str
    modelLang: str
    modelSpacyVersion: str
    modelSpacyGitVersion: str


# UIMA type: mark modification of the document
class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# Lemma
class Lemma(BaseModel):
    begin: int
    end: int
    lemma: str
    write: bool

# Response of this annotator
# Note, this is transformed by the Lua script
class TextImagerResponse(BaseModel):
    lemmas: List[Lemma]
    # Annotation meta, containing model name, version and more
    # Note: Same for each annotation, so only returned once
    meta: Optional[AnnotationMeta]
    # Modification meta, one per document
    modification_meta: Optional[DocumentModification]


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


# Create LRU cache with max size for model
# lru_cache_with_size = lru_cache(maxsize=3)
lru_cache_tagger_with_size = lru_cache(maxsize=3)

# Lock for model loading
model_load_lock = Lock()
tagger_load_lock = Lock()

@lru_cache_tagger_with_size
def load_cache_tagger(model_name):
    logger.info("Loading diaparser model \"%s\"...", model_name)
    tagger = load_tagger(model_name)

    logger.info("Finished loading diaparser model \"%s\"", model_name)
    return tagger

def load_tagger(model_name):
    model_load_lock.acquire()
    tagger = ht.HanoverTagger(model_name)
    model_load_lock.release()

    return tagger

def get_tagger_model_name():
    # Directly use specified in parameters, this ignores any model variants!
    model_name = os.environ['TEXTIMAGER_HANTA_MODEL_NAME']

    if model_name == '':
        model_name = "morphmodel_ger.pgz"
#     logger.debug("Mapped model name from document language \"%s\": \"%s\"", document_lang, model_name)
    return model_name

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemHANTA.xml'
# logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
#     logger.debug("Base typesystem:")
#     logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "textimager_duui_hanta.lua"
# logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
#     logger.debug("Lua communication script:")
#     logger.debug(lua_communication_script_filename)


# Start fastapi
# TODO openapi types are not shown?
# TODO self host swagger files: https://fastapi.tiangolo.com/advanced/extending-openapi/#self-hosting-javascript-and-css-for-docs
app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.textimager_hanta_annotator_name,
    description="HantALemmatizer implementation for TTLab TextImager DUUI",
    version=settings.textimager_hanta_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "konca@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=sorted(list(SPACY_SUPPORTED_LANGS)),
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.textimager_hanta_annotator_name,
        version=settings.textimager_hanta_annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            #"spacy_version": spacy.__version__
        },
        # TODO
        docker_container_id="[TODO]",
        parameters={
            # Write the following types, if empty/null all are written
            # Note: All data is always generated by the full document text, dependency is only written if tokens are written too
            "write_types": TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES,
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    # TODO remove cassis dependency, as only needed for typesystem at the moment?
    xml = typesystem.to_xml()

    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


dt = datetime.now()

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    dt = datetime.now()
    tokens = request.tokens
    text = request.text

    tagger = load_cache_tagger(settings.textimager_hanta_model_name)
    lemmas = []

    print(dt, f'Processing {len(tokens)} sentences', end=' ')


    for sentence in tokens:
        hanta = tagger.tag_sent([t["text"] for t in sentence])
        assert len(hanta) == len(sentence)
        for i, l in enumerate(hanta):
            current_lemma = Lemma(
                begin=sentence[i]["begin"],
                end=sentence[i]["end"],
                lemma=hanta[i][1],
                write=True
            )
            lemmas.append(current_lemma)

    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)


    return TextImagerResponse(
        lemmas=lemmas,
        # meta=meta,
        # modification_meta=modification_meta
    )
