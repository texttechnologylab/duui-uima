import logging
from collections import defaultdict
from typing import List, Union

import yaml
from cassis import load_typesystem
from fastapi import FastAPI, Response
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import PlainTextResponse


class Config(BaseSettings):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    annotator_version: str
    # Log level
    log_level: str


class UimaCategoryCoveredTagsAnnotation(BaseModel):
    value: str
    score: float
    tags: str
    begin: int
    end: int


class UimaTopic(BaseModel):
    value: str
    score: float


class UimaTopicAnnotation(BaseModel):
    topics: List[UimaTopic]
    begin: int
    end: int


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    doc_len: int
    doc_lang: str
    doc_text: str
    anns: Union[List[UimaTopicAnnotation], List[UimaCategoryCoveredTagsAnnotation]]
    type: str  # type of the annotation


# Response sent by DUUI
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    begin: List[int]
    end: List[int]
    results: List
    len_results: List[int]
    factors: List
    model_name: Union[None, str]
    model_version: Union[None, str]
    model_lang: Union[None, str]
    model_source: Union[None, str]


def process_category_covered_tags_anns(annotations):
    begin = []
    end = []
    idx2results = defaultdict(list)
    idx2scores = defaultdict(list)

    for doc_idx, ann in enumerate(annotations):

        begin_i = ann.begin
        end_i = ann.end

        if begin_i not in begin:
            begin.append(begin_i)

        if end_i not in end:
            end.append(end_i)

        idx2results[(begin_i, end_i)].append(ann.value)
        idx2scores[(begin_i, end_i)].append(ann.score)

    output = {
        "begin": begin,
        "end": end,
        "len_results": [len(v) for k, v in idx2results.items()],
        "results": list(idx2results.values()),
        "scores": list(idx2scores.values()),

    }

    return output


def process_transformer_topic_anns(annotations):
    begin = []
    end = []
    results = []
    factors = []
    len_results = []

    for doc_idx, ann in enumerate(annotations):

        begin_i = ann.begin
        end_i = ann.end
        begin.append(begin_i)
        end.append(end_i)

        r = []
        f = []

        for topic in ann.topics:
            r.append(topic.value)
            f.append(topic.score)
        results.append(r)
        factors.append(f)
        len_results.append(len(r))

    output = {
        "begin": begin,
        "end": end,
        "len_results": len_results,
        "results": results,
        "scores": factors,

    }

    return output


config_file = 'config.yaml'
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

config = Config(annotator_name=config['annotation']['name'], annotator_version=str(config['annotation']['version']),
                log_level=config['log_level'], )

logging.basicConfig(level=config.log_level)
logger = logging.getLogger(__name__)

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemUnifiedTopic.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=config.annotator_name,
    description="Factuality annotator",
    version=config.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "bagci@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Load the Lua communication script
lua_communication_script_filename = "duui-tts-converter.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)

with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
logger.debug("Lua communication script:")
logger.debug(lua_communication_script_filename)


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


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Return documentation info
@app.get("/v1/documentation")
def get_documentation():
    return "Test"


@app.post("/v1/process")
def post_process(request: DUUIRequest):
    ann_type = request.type.split('.')[-1]
    begin = []
    end = []
    len_results = []
    results = []
    scores = []
    model_src = None
    model_version = None
    model_name = None
    model_lang = request.doc_lang
    if "tags" in request:
        tags = request.anns[0].tags
        model_src = tags.split(';')[-1]
        model_version = '_'.join(model_src.split(';')[:2])
        model_name = model_src.split(';')[0]

    try:
        if ann_type == 'CategoryCoveredTagged':
            processed_sentences = process_category_covered_tags_anns(request.anns)
        if ann_type == 'Topic' or ann_type == 'BertTopic':
            processed_sentences = process_transformer_topic_anns(request.anns)
        begin = begin + processed_sentences["begin"]
        end = end + processed_sentences["end"]
        len_results = len_results + processed_sentences["len_results"]
        results = results + processed_sentences["results"]
        scores = scores + processed_sentences["scores"]

    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(begin=begin, end=end, results=results,
                        len_results=len_results, model_name=model_name,
                        model_version=model_version, model_source=model_src,
                        model_lang=model_lang, factors=scores)
