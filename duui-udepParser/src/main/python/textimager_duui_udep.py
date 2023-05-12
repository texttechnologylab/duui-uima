# import bfsrl
# import func_timeout
import logging
import os
import torch

from cassis import load_typesystem
from datetime import datetime
from diaparser.parsers import Parser
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
    textimager_udepparser_annotator_name: str
    # Version of this annotator
    textimager_udepparser_annotator_version: str
    # Log level
    textimager_udepparser_log_level: str
    # Model name
    textimager_udepparser_model_name: str
    # Diaparser batch size
    textimager_udepparser_batch_size: int


# Load settings from env vars
settings = Settings()

# Init logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=settings.textimager_udepparser_log_level,#logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(settings.textimager_udepparser_log_level)
logger.info("TTLab TextImager DUUI udepParser")
logger.info("Name: %s", settings.textimager_udepparser_annotator_name)
logger.info("Version: %s", settings.textimager_udepparser_annotator_version)

# Type names needed by this annotator
UIMA_TYPE_DEPENDENCY = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
# Note: Without any extra meta types or subtypes
TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    UIMA_TYPE_DEPENDENCY,
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
    # spacyVersion: str
    modelLang: str
    # modelSpacyVersion: str
    # modelSpacyGitVersion: str


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
    token_ind: int
    write: bool


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
    tokens: List[Token]
    udeps: List[Dependency]
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
lru_cache_diaparser_with_size = lru_cache(maxsize=3)

# Lock for model loading
model_load_lock = Lock()
diaparser_load_lock = Lock()

def load_parser(device, model_name='de_hdt.dbmdz-bert-base'):
    args = {'use_multiprocessing': False, 'use:multiprocessing_for_evaluation': False,
            'process_count': 1, 'verbose': True}
    #print('USING', f'{"GPU" if torch.cuda.is_available() else "CPU"}')
    return Parser.load(model_name, args=args, device=device)

@lru_cache_diaparser_with_size
def load_cache_diaparser_model(model_name, device):
    logger.info("Loading diaparser model \"%s\"...", model_name)
    parser = load_parser(device, diaparser_model_name)
    logger.info("Finished loading diaparser model \"%s\"", model_name)
    return parser

# Get spaCy model from language
def get_parser_model_name():
    # Directly use specified in parameters, this ignores any model variants!
    model_name = os.environ['TEXTIMAGER_UDEPPARSER_MODEL_NAME']

    if model_name == '':
        model_name = "de_hdt.dbmdz-bert-base"
#     logger.debug("Mapped model name from document language \"%s\": \"%s\"", document_lang, model_name)
    return model_name

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemUDEP.xml'
# logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
#     logger.debug("Base typesystem:")
#     logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "textimager_duui_udep.lua"
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
    title=settings.textimager_udepparser_annotator_name,
    description="Universal Dependency Parser implementation for TTLab TextImager DUUI",
    version=settings.textimager_udepparser_annotator_version,
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
        annotator_name=settings.textimager_udepparser_annotator_name,
        version=settings.textimager_udepparser_annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
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

diaparser_model_name = get_parser_model_name()

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

    udeps = []
    tokens_out = []

    parser = load_cache_diaparser_model(diaparser_model_name, device)

    dt = datetime.now()
    tokens__ = request.tokens
    text = request.text

    print(dt, f'Processing {len(tokens__)} sentences', end=' ')

    try:
        batch_size = int(os.environ['TEXTIMAGER_UDEPPARSER_BATCH_SIZE'])
    except ValueError:
        print('Batch size not set, using default (256)')
        batch_size = 256
    print('using batch size', batch_size, flush=True)

    tokens_ = []
    for i, ts in enumerate(tokens__):
        if len(ts) == 0 or len(ts) > 150:
            continue
        tokens_.append(ts)


    count_token = 0
    for i in range(0, len(tokens_), batch_size):
        if device == 'GPU':
            logger.info('emptying cuda cache')
            torch.cuda.empty_cache()
            logger.info('cuda cache empty')

        tokens_batch = [ts for ts in tokens_[i: i+batch_size]]

        ts_batch = []
        for tokens in tokens_batch:
            ts_batch.append([t['text'] for t in tokens])

        try:
            diaparse = parser.predict(ts_batch)
        except IndexError as e:
            #print(ts_batch)
            logger.debug('Diaparser IndexError')
            logger.debug(e)
            continue

        assert len(diaparse.sentences) == len(tokens_batch)

        for j in range(len(tokens_batch)):
            assert len(diaparse.sentences[j]) == len(tokens_batch[j])


            # print(tokens_batch[j])
            # print(diaparse.sentences[j].rels)
            # print(diaparse.sentences[j].values[6])
            for k in range(len(tokens_batch[j])):
                # print(tokens_batch[j][k])
                # print(diaparse.sentences[j].rels[k])
                token_begin = tokens_batch[j][k]['begin']
                token_end = tokens_batch[j][k]['end']
                # token_text = tokens_batch[j][k]['text']
                token_head = diaparse.sentences[j].values[6][k]
                token_head = k if token_head == 0 else token_head - 1
                # print(token_begin, token_end, token_head, token_text)
                current_dep = Dependency(
                    begin=token_begin,
                    end=token_end,
                    type=diaparse.sentences[j].rels[k],
                    flavor='udep',
                    dependent_ind=k,
                    governor_ind=token_head,
                    token_ind=count_token,
                    write=True
                )


                current_token = Token(
                    begin=token_begin,
                    end=token_end,
                    ind=count_token
                )

                assert len(udeps) == count_token
                udeps.append(current_dep)
                tokens_out.append(current_token)
                count_token += 1
            # print()

        logger.info(f'{min(i+1+batch_size, len(tokens_))}/{len(tokens_)} done')


    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)

    if device == 'GPU':
        logger.info('emptying cuda cache')
        torch.cuda.empty_cache()
        logger.info('cuda cache empty')

    assert len(tokens_out) == len(udeps)
    return TextImagerResponse(
        tokens=tokens_out,
        udeps=udeps,
        meta=meta,
        modification_meta=modification_meta
    )
