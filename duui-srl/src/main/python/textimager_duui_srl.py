import srl
# import func_timeout
import logging
import os
import sys
import torch

from cassis import load_typesystem
from datetime import datetime
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from functools import lru_cache
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
    textimager_srl_annotator_name: str
    # Version of this annotator
    textimager_srl_annotator_version: str
    # Log level
    textimager_srl_log_level: str
    # Model type
    textimager_srl_parser_model_type: str
    # Model name
    textimager_srl_parser_model_name: str
    #batch size
    textimager_srl_parser_batch_size: int


# sys.path.append('crfsrl')
# Load settings from env vars
settings = Settings()

# Init logger
# logging.basicConfig(level=settings.textimager_bfsrl_log_level)
# logging.basicConfig(
#     format='%(asctime)s %(levelname)-8s %(message)s',
#     level=settings.textimager_srl_log_level,#logging.INFO,
#     datefmt='%Y-%m-%d %H:%M:%S')

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=settings.textimager_srl_log_level,#logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(settings.textimager_srl_log_level)
logger.info("TTLab TextImager DUUI SRL")
logger.info("Name: %s", settings.textimager_srl_annotator_name)
logger.info("Version: %s", settings.textimager_srl_annotator_version)

# Type names needed by this annotator
UIMA_TYPE_SENTITY = "org.texttechnologylab.annotation.semaf.isobase.Entity"
UIMA_TYPE_SRLINK = "rg.texttechnologylab.annotation.semaf.semafsr.SrLink"

# Note: Without any extra meta types or subtypes
TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    UIMA_TYPE_SENTITY,
    UIMA_TYPE_SRLINK
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


# Token
class Token(BaseModel):
    begin: int
    end: int
    ind: int
    write_token: Optional[bool]
    lemma: Optional[str]
    write_lemma: Optional[bool]
    pos: Optional[str]
    pos_coarse: Optional[str]
    write_pos: Optional[bool]
    morph: Optional[str]
    morph_details: Optional[dict]
    write_morph: Optional[bool]
    parent_ind: Optional[int]
    write_dep: Optional[bool]


# Dependency
class Dependency(BaseModel):
    begin: int
    end: int
    type: str
    flavor: str
    dependent_ind: int
    governor_ind: int
    write_dep: bool


# Entity
class Entity(BaseModel):
    begin: int
    end: int
    write: bool

# Link
class Link(BaseModel):
    begin_fig: int
    end_fig: int
    begin_gr: int
    end_gr: int
    rel_type: str
    write: bool


# Response of this annotator
# Note, this is transformed by the Lua script
class TextImagerResponse(BaseModel):
    entities: List[Entity]
    links: List[Link]
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
# lru_cache_with_size = lru_cache(maxsize=settings.textimager_spacy_model_cache_size)
lru_cache_parser_with_size = lru_cache(maxsize=3)

# Lock for model loading
model_load_lock = Lock()

@lru_cache_parser_with_size
def load_cache_parser_model(device, model_name, model_type):
    logger.info("Loading diaparser model \"%s\"...", model_name)
    parser, args = srl.load_parser(device, model_name, model_type)
    logger.info("Finished loading diaparser model \"%s\"", model_name)

    return parser, args

# def load_parser(model_name, model_type):
#     model_load_lock.acquire()
#
#     err = None
#     try:
#         logger.info(f'Getting {model_name}')
#         parser = load_cache_parser_model(model_name, model_type)
#     except Exception as e:
#         parser = None
#         err = str(e)
#         logging.exception(f'Failed to load model: {e}')
#
#     model_load_lock.release()
#
#
#     return parser, err

# Get spaCy model from language
def get_parser_model_name():
    # Directly use specified in parameters, this ignores any model variants!
    model_name = os.environ['TEXTIMAGER_SRL_PARSER_MODEL_NAME']
    model_type = os.environ['TEXTIMAGER_SRL_PARSER_MODEL_TYPE']

    if model_name == '':
        model_name = 'exp/xlm_roberta_base_de/model'
        model_type = 'xlm_roberta_base'
#     logger.debug("Mapped model name from document language \"%s\": \"%s\"", document_lang, model_name)
    return model_name, model_type

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemSRL.xml'
# logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
#     logger.debug("Base typesystem:")
#     logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "textimager_duui_srl.lua"
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
    title=settings.textimager_srl_annotator_name,
    description="BFSRL implementation for TTLab TextImager DUUI",
    version=settings.textimager_srl_annotator_version,
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
        annotator_name=settings.textimager_srl_annotator_name,
        version=settings.textimager_srl_annotator_version,
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
#print(dt, 'loading tagger', flush=True)
# tagger = srl.load_tagger()

# parser_model_name = get_parser_model_name()
model_name, model_type = get_parser_model_name()
# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    logger.info(f'USING {device}')
    if device == 'GPU':
        logger.info('emptying cuda cache')
        torch.cuda.empty_cache()
        logger.info('cuda cache empty')

    # Return data
    meta = None
    modification_meta = None



    dt = datetime.now()
    tokens__ = request.tokens
    text = request.text

    print(dt, f'Processing {len(tokens__)} sentences')
    #print('-'*30, settings.textimager_srl_parser_model_name)
    parser, args = load_cache_parser_model(device, model_name, model_type)
    # tokens__ = srl.predict_roles(tokens__, parser, model_path=settings.textimager_srl_parser_model_name,
    #                              model_type=settings.textimager_srl_parser_model_type)
    args.batch_size = settings.textimager_srl_parser_batch_size
    tokens__ = srl.predict_roles(tokens__, parser, args)

    entities = {}
    links = []
    done = set()
    for sentence in tokens__:
        preds_done = {}
        for i, token in enumerate(sentence, start=1):
            try:
                args = token['arg'].split('|')
            except KeyError:
                continue
            for j in range(len(args)):
                split = args[j].split(':')
                if split[0] == '_'  or len(split) < 2:
                    continue
                token_begin = int(token['begin'])
                token_end = int(token['end'])
                try:
                    token_i = int(split[0])
                except ValueError:
                    continue
                if token_i == 0:
                    if (token_begin, token_end) in entities:
                        entity = entities[(token_begin, token_end)]
                    else:
                        entity = Entity(begin=token_begin, end=token_end, write=True)
                        entities[(token_begin, token_end)] = entity
                    # entity = Entity(begin=token_begin, end=token_end, write=True)
                    # entities[(token_begin, token_end)] = entity
                    preds_done[i] = (token_begin, token_end)
        for i, token in enumerate(sentence, start=0):
            try:
                args = token['arg'].split('|')
            except KeyError:
                continue
            for j in range(len(args)):
                split = args[j].split(':')#.replace('B-', '').replace('I-', '')
                if split[0] == '_'  or len(split) < 2:
                    continue
                token_arg = split[1]
                token_begin = token['begin']
                token_end = token['end']
                token_i = int(split[0])
                if token_i != 0:
                    pred_begin, pred_end = preds_done[token_i]
                    if (pred_begin, pred_end, token_begin, token_end) in done:
                        continue
                    if (token_begin, token_end) in entities:
                        entity = entities[(token_begin, token_end)]
                    else:
                        entity = Entity(begin=token_begin, end=token_end, write=True)
                        entities[(token_begin, token_end)] = entity
                    links.append(Link(begin_fig=pred_begin, end_fig=pred_end,
                        begin_gr=token_begin, end_gr=token_end,
                        rel_type=token_arg, write=True))
                    done.add((pred_begin, pred_end, token_begin, token_end))

    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)

    if device == 'GPU':
        logger.info('emptying cuda cache')
        torch.cuda.empty_cache()
        logger.info('cuda cache empty')

    return TextImagerResponse(
        entities=[x for _, x in entities.items()],
        links=links,
        meta=meta,
        modification_meta=modification_meta
    )

#if __main_
