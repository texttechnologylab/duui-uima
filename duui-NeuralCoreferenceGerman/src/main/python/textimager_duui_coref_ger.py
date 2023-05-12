import logging
from datetime import datetime
from platform import python_version
from sys import version as sys_version

from cassis import load_typesystem
import torch

from typing import List, Optional, Dict
from pydantic import BaseSettings, BaseModel
from functools import lru_cache

from threading import Lock

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse

from ModelHandler import CorefHandler

import os
#os.environ["TEXTIMAGER_COREF_ANNOTATOR_NAME"] = "coref_cuda"
#os.environ["TEXTIMAGER_COREF_ANNOTATOR_VERSION"] = "0.0.1"
#os.environ["TEXTIMAGER_COREF_LOG_LEVEL"] = "DEBUG"
#os.environ["TEXTIMAGER_COREF_PARSER_MODEL_NAME"] = "se10_electra_uncased"
#os.environ["TEXTIMAGER_COREF_PARSER_MODEL_NAME"] = "droc_incremental_no_features_no_segment_distance"

# Settings
# These are automatically loaded from env variables
class Settings(BaseSettings):
    # Name of this annotator
    textimager_coref_annotator_name: str
    # Version of this annotator
    textimager_coref_annotator_version: str
    # Log level
    textimager_coref_log_level: str
    # Model name
    textimager_coref_parser_model_name: str
    # Model LRU cache size


settings = Settings()

# Init logger
# logging.basicConfig(level=settings.textimager_bfsrl_log_level)
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=settings.textimager_coref_log_level,
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Coref German")
logger.info("Name: %s", settings.textimager_coref_annotator_name)
logger.info("Version: %s", settings.textimager_coref_annotator_version)

# Type names needed by this annotator
UIMA_TYPE_SENTITY = "org.texttechnologylab.annotation.semaf.isobase.Entity"
UIMA_TYPE_META = "org.texttechnologylab.annotation.semaf.meta.MetaLink"

# Note: Without any extra meta types or subtypes
TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    UIMA_TYPE_SENTITY,
    UIMA_TYPE_META
]


class TextImagerRequest(BaseModel):
    text: str
    lang: str
    tokenized_document: List[List[Dict]]
    parameters: Optional[dict]


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str
    modelLang: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


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


class TextImagerResponse(BaseModel):
    entities: List[Entity]
    links: List[Link]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]


class TextImagerCapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


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


lru_cache_diaparser_with_size = lru_cache(maxsize=3)

model_load_lock = Lock()
diaparser_load_lock = Lock()


@lru_cache_diaparser_with_size
def load_cache_diaparser_model(model_name):
    logger.info("Loading diaparser model \"%s\"...", model_name)
    parser = CorefHandler()
    logger.info("Finished loading diaparser model \"%s\"", model_name)
    return parser


def get_parser_model_name():
    # TODO: for now, we only support one model
    #model_name = "tuba10_electra_uncased"
    model_name = "tuba10_electra_gelectra"
    return model_name


typesystem_filename = 'TypeSystemCoref.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    print(typesystem.to_xml())


lua_communication_script_filename = "textimager_duui_coref_ger.lua"
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.textimager_coref_annotator_name,
    description="Coreference implementation for TTLab TextImager DUUI",
    version=settings.textimager_coref_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "henlein@em.uni-frankfurt.de",
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
        supported_languages=sorted(list(["de", "ger"])),
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.textimager_coref_annotator_name,
        version=settings.textimager_coref_annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
        },
        # TODO
        docker_container_id="[TODO]",
        parameters={
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

handler = CorefHandler()

@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    print("post_process")
    print(request)
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

    dt = datetime.now()
    tokenized_document = request.tokenized_document
    text = request.text
    print("tokenized_document", tokenized_document)
    token_list = []
    flat_token_list = []
    for tokenized_sent in tokenized_document:
        token_list.append([token["text"] for token in tokenized_sent])
        flat_token_list.extend(tokenized_sent)
    print("token_list", token_list)
    #flat_token_list = [item for sublist in token_list for item in sublist]
    print("flat_token_list", flat_token_list)

    preprocessed = handler.preprocess(token_list)
    print("preprocessed", preprocessed)
    inference_output = handler.inference(preprocessed)
    print("inference_output", inference_output)
    postprocessed = handler.postprocess(inference_output)
    print("postprocessed", postprocessed)
    postprocessed = postprocessed[0]

    if len(postprocessed) == 0:
        return TextImagerResponse(
            meta=meta,
            modification_meta=modification_meta,
            entities=[],
            links=[],
        )

    entities = []
    links = []
    for cluster in postprocessed:
        ground = cluster[0]
        #print("==============")
        #print(ground)
        #print(flat_token_list)
        #print(flat_token_list[ground[0]])
        g_begin = flat_token_list[ground[0]]["begin"]
        g_end = flat_token_list[ground[1]]["end"]
        entities.append(Entity(begin=g_begin, end=g_end, write=True))

        for figure in cluster[1:]:
            f_begin = flat_token_list[figure[0]]["begin"]
            f_end = flat_token_list[figure[1]]["end"]
            entities.append(Entity(begin=f_begin, end=f_end, write=True))
            links.append(Link(begin_fig=f_begin, end_fig=f_end,
                              begin_gr=g_begin, end_gr=g_end,
                              rel_type="COREF", write=True))

    print(postprocessed)

    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)

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


if __name__ == "__main__":
    uvicorn.run(app)
