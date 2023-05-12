import bfsrl
import func_timeout
import logging
import os
import spacy
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
    textimager_bfsrl_annotator_name: str
    # Version of this annotator
    textimager_bfsrl_annotator_version: str
    # Log level
    textimager_bfsrl_log_level: str
    # Model name
    textimager_bfsrl_parser_model_name: str
    # Model LRU cache size
    textimager_spacy_model_cache_size: int
    # Diaparser atch size
    textimager_diaparser_batch_size: int


# Load settings from env vars
settings = Settings()

# Init logger
# logging.basicConfig(level=settings.textimager_bfsrl_log_level)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=settings.textimager_bfsrl_log_level,#logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI BFSRL")
logger.info("Name: %s", settings.textimager_bfsrl_annotator_name)
logger.info("Version: %s", settings.textimager_bfsrl_annotator_version)

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
lru_cache_with_size = lru_cache(maxsize=settings.textimager_spacy_model_cache_size)
lru_cache_diaparser_with_size = lru_cache(maxsize=3)

# Lock for model loading
model_load_lock = Lock()
diaparser_load_lock = Lock()

@lru_cache_diaparser_with_size
def load_cache_diaparser_model(model_name):
    logger.info("Loading diaparser model \"%s\"...", model_name)
    parser = bfsrl.load_parser(diaparser_model_name)
    logger.info("Finished loading diaparser model \"%s\"", model_name)
    return parser

spacy_model_name = "de_core_news_lg"

@lru_cache_with_size
def load_cache_spacy_model(model_name):
    logger.info("Loading spaCy model \"%s\"...", model_name)
    nlp = spacy.load(model_name)
    logger.info("Finished loading spaCy model \"%s\"", model_name)
    return nlp


# Load spaCy model using LRU cached function
def load_spacy_model(model_name):
    model_load_lock.acquire()

    err = None
    try:
        logger.info("Getting spaCy model \"%s\"...", model_name)
        nlp = load_cache_spacy_model(model_name)
    except Exception as ex:
        nlp = None
        err = str(ex)
        logging.exception("Failed to load spaCy model: %s", ex)

    model_load_lock.release()


    return nlp, err

# Get spaCy model from language
def get_parser_model_name():
    # Directly use specified in parameters, this ignores any model variants!
    model_name = os.environ['TEXTIMAGER_BFSRL_PARSER_MODEL_NAME']

    if model_name == '':
        model_name = "de_hdt.dbmdz-bert-base"
#     logger.debug("Mapped model name from document language \"%s\": \"%s\"", document_lang, model_name)
    return model_name

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemBFSRL.xml'
# logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
#     logger.debug("Base typesystem:")
#     logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "textimager_duui_bfsrl.lua"
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
    title=settings.textimager_bfsrl_annotator_name,
    description="BFSRL implementation for TTLab TextImager DUUI",
    version=settings.textimager_bfsrl_annotator_version,
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
        annotator_name=settings.textimager_bfsrl_annotator_name,
        version=settings.textimager_bfsrl_annotator_version,
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
# tagger = bfsrl.load_tagger()

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


    def signal_handler(signum, frame):
        raise Exception("Timed out!")

    count_no_verbs = 0

    parser = load_cache_diaparser_model(diaparser_model_name)
#     parser = bfsrl.load_parser(diaparser_model_name)


    # Load model, this is cached
    nlp, nlp_err = load_spacy_model(spacy_model_name)
    if nlp is None:
        raise Exception(f"spaCy model \"{model_name}\" could not be loaded: {nlp_err}")

    entities = []
    links = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())


    dt = datetime.now()
    tokens__ = request.tokens
    text = request.text

    print(dt, f'Processing {len(tokens__)} sentences', end=' ')

    processed_preds = set()
    try:
        batch_size = int(os.environ['TEXTIMAGER_DIAPARSER_BATCH_SIZE'])
    except ValueError:
        print('Batch size not set, using default (256)')
        batch_size = 256
    print('using batch size', batch_size, flush=True)

    tokens_ = []
    for i, ts in enumerate(tokens__):
        if len(ts) == 0 or len(ts) > 150:
            continue
        poss = []
        for t in ts:
            try:
                poss.append(t['pos'])
            except KeyError:
                poss.append('X')
        if not 'VERB' in poss and not 'AUX' in poss:
            count_no_verbs += 1
            continue

        tokens_.append(ts)

    logger.info(f'number of sentences with VERB/AUX {len(tokens_)}')

    for i in range(0, len(tokens_), batch_size):
        if device == 'GPU':
            logger.info('emptying cuda cache')
            torch.cuda.empty_cache()
            logger.info('cuda cache empty')

        tokens_batch = [ts for ts in tokens_[i: i+batch_size]]

        ts_batch = []
        #poss_batch = []
        #tags_batch = []
        import sys
        for tokens in tokens_batch:
            ts_batch.append([t['text'] for t in tokens])
            #poss_batch.append([t['pos'] for t in tokens])
            #tags_batch.append([t['tag'] for t in tokens])

        try:
            diaparse = parser.predict(ts_batch)
        except IndexError as e:
            print(ts_batch)
            logger.debug('Diaparser IndexError')
            logger.debug(e)
            continue

        assert len(diaparse.sentences) == len(tokens_batch)

        for j, tokens in enumerate(tokens_batch):
            ts = [t['text'] for t in tokens]
            ts_begin = [int(t['begin']) for t in tokens]
            ts_end = [int(t['end']) for t in tokens]
            poss = [t['pos'] for t in tokens]
            tags = [t['tag'] for t in tokens]
            deps = [t['dep'] for t in tokens]
            assert len(ts) == len(tokens)
            assert len(ts) == len(poss)
            assert len(ts) == len(tags)
            assert len(ts) == len(ts_begin)
            assert len(ts) == len(ts_end)
            assert len(ts) == len(deps)

            try:
                results = func_timeout.func_timeout(10, bfsrl.srl, (diaparse.sentences[j], ts, poss, tags,
                 nlp, 'all', False, logger))
            except func_timeout.FunctionTimedOut:
                logger.debug('Timed out!')
                results = None

            if results is None:
                continue
            words, ts_, tags, udeps, udeps_heads, psrs = results

#             logger.info(f'number of extracted items {len(psrs)}')
            for tri in psrs:
                try:
                    pred_i = int(tri[-1]['i']) - 1
                    if not (ts_begin[pred_i], ts_end[pred_i]) in processed_preds:
                        entities.append(Entity(begin=ts_begin[pred_i], end=ts_end[pred_i], write=True))
                        processed_preds.add((ts_begin[pred_i], ts_end[pred_i]))
                except KeyError:
                    continue
                #ARGO
                try:
                    arg_is = tri[0]['i']
                    if not isinstance(arg_is, list):
                        arg_is = [int(arg_is) - 1]
                    else:
                        arg_is = [int(x) - 1 for x in arg_is]
                    for arg_i in arg_is:
                        entities.append(Entity(begin=ts_begin[arg_i], end=ts_end[arg_i], write=True))
                        links.append(Link(begin_fig=ts_begin[pred_i], end_fig=ts_end[pred_i],
                            begin_gr=ts_begin[arg_i], end_gr=ts_end[arg_i],
                            rel_type='ARG0', write=True))
                except KeyError:
                    pass
                #ARG1
                try:
                    arg_is = tri[1]['i']
                    if not isinstance(arg_is, list):
                        arg_is = [int(arg_is) - 1]
                    else:
                        arg_is = [int(x) - 1 for x in arg_is]
                    for arg_i in arg_is:
                        entities.append(Entity(begin=ts_begin[arg_i], end=ts_end[arg_i], write=True))
                        links.append(Link(begin_fig=ts_begin[pred_i], end_fig=ts_end[pred_i],
                            begin_gr=ts_begin[arg_i], end_gr=ts_end[arg_i],
                            rel_type='ARG1', write=True))
                except KeyError:
                    pass
                #ARG2
                try:
                    arg_is = tri[2]['i']
                    if not isinstance(arg_is, list):
                        arg_is = [int(arg_is) - 1]
                    else:
                        arg_is = [int(x) - 1 for x in arg_is]
                    for arg_i in arg_is:
                        entities.append(Entity(begin=ts_begin[arg_i], end=ts_end[arg_i], write=True))
                        links.append(Link(begin_fig=ts_begin[pred_i], end_fig=ts_end[pred_i],
                            begin_gr=ts_begin[arg_i], end_gr=ts_end[arg_i],
                            rel_type='ARG2', write=True))
                except KeyError:
                    pass

                try:
                    arg_is = tri[5]['i']
                    if not isinstance(arg_is, list):
                        arg_is = [int(arg_is) - 1]
                    else:
                        arg_is = [int(x) - 1 for x in arg_is]
                    for arg_i in arg_is:
                        entities.append(Entity(begin=ts_begin[arg_i], end=ts_end[arg_i],
                            write=True))
                        links.append(Link(begin_fig=ts_begin[pred_i],
                            end_fig=ts_end[pred_i], begin_gr=ts_begin[arg_i],
                            end_gr=ts_end[arg_i],
                            rel_type='VG', write=True))
                except KeyError:
                    pass
                try:
                    arg_is = tri[6]['i']
                    arg_role = tri[6]['role']
                    if not isinstance(arg_is, list):
                        arg_is = [int(arg_is) - 1]
                    else:
                        arg_is = [int(x) - 1 for x in arg_is]
                    for arg_i in arg_is:
                        entities.append(Entity(begin=ts_begin[arg_i], end=ts_end[arg_i],
                            write=True))
                        links.append(Link(begin_fig=ts_begin[pred_i],
                            end_fig=ts_end[pred_i],
                            begin_gr=ts_begin[arg_i], end_gr=ts_end[arg_i],
                            rel_type=arg_role, write=True))
                            #rel_type='ARGM', write=True))
                except KeyError:
                    pass
        logger.info(f'{min(i+1+batch_size, len(tokens_))}/{len(tokens_)} done')
        logger.info(f'{min(i+1+batch_size, len(tokens_))}/{len(tokens_)} done')


    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)
    print('Number of sentences without a VERB:', count_no_verbs, flush=True)

    if device == 'GPU':
        logger.info('emptying cuda cache')
        torch.cuda.empty_cache()
        logger.info('cuda cache empty')

    return TextImagerResponse(
        entities=entities,
        links=links,
        meta=meta,
        modification_meta=modification_meta
    )
