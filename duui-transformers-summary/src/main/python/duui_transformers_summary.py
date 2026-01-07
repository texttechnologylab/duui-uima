from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
import torch
from threading import Lock
from functools import lru_cache
from summarization import Summarization, MDMLSummarization, MT5Summarization, PegasusSummarization
import numpy as np

sources = {
    "MT5": "https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum",
    "MDML": "https://github.com/airKlizz/mdmls",
    "Google T5": "https://huggingface.co/google/flan-t5-base",
    "Pegasus Financial": "https://huggingface.co/human-centered-summarization/financial-summarization-pegasus",
}
languages = {
    "MT5": "Multi",
    "MDML": "Multi",
    "Google T5": "Multi",
    "Pegasus Financial": "English",
}
versions = {
    "MT5": "2437a524effdbadc327ced84595508f1e32025b3",
    "MDML": "60f9eadb55d20eae889332035daa884205971566",
    "Google T5": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
    "Pegasus Financial": "734fe2da8db6e4d7272ad553cb3343ed59a566d7",
}
# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse
model_lock = Lock()
device = 0 if torch.cuda.is_available() else "cpu"

class Settings(BaseSettings):
    # Name of this annotator
    summary_annotator_name: str
    # Version of this annotator
    summary_annotator_version: str
    # Log level
    summary_log_level: str
    # # model_name
    # summary_model_name: str
    # Name of this annotator
    # summary_model_version: str
    #cach_size
    summary_model_cache_size: int


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.summary_model_cache_size)
logging.basicConfig(level=settings.summary_log_level)
logger = logging.getLogger(__name__)

device = 0 if torch.cuda.is_available() else "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemSummary.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_summary.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    #
    all_annotations: List[Dict]
    #
    model_name: str
    #
    lang: str
    #
    summary_length: int


# UIMA type: mark modification of the document
class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# UIMA type: adds metadata to each annotation
class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


# Response sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerResponse(BaseModel):
    # Symspelloutput
    # List of Sentence with every token
    # Every token is a dictionary with following Infos:
    # Symspelloutput right if the token is correct, wrong if the token is incorrect, skipped if the token was skipped, unkownn if token can corrected with Symspell
    # If token is unkown it will be predicted with BERT Three output pos:
    # 1. Best Prediction with BERT MASKED
    # 2. Best Cos-sim with Sentence-Bert and with perdicted words of BERT MASK
    # 3. Option 1 and 2 together
    meta: AnnotationMeta
    # Modification meta, one per document
    modification_meta: DocumentModification
    begin: List[int]
    end: List[int]
    summaries: List[str]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.summary_annotator_name,
    description="Factuality annotator",
    version=settings.summary_annotator_version,
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

@lru_cache_with_size
def load_model(model_name):
    match model_name:
        case "MDML":
            if device == "cpu":
                device_in = -1
            else:
                device_in = device
            model_i = MDMLSummarization(device_in)
        case "MT5":
            model_i = MT5Summarization("csebuetnlp/mT5_multilingual_XLSum", device)
        case "Google T5":
            model_i = Summarization("google/flan-t5-base", device)
        case "Pegasus Financial":
            model_i = PegasusSummarization("human-centered-summarization/financial-summarization-pegasus", device)
    return model_i


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text

def process_selection(model_name, sentences, sum_len):
    output = {
        "begin": [],
        "end": [],
        "summary": [],
    }
    with model_lock:
        model_i = load_model(model_name)
        for sentence in sentences:
            text = sentence["text"]
            begin = sentence["begin"]
            end = sentence["end"]
            summary = model_i.summarize(text, sum_len)
            output["begin"].append(begin)
            output["end"].append(end)
            output["summary"].append(summary)
    return output

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):
    # Return data
    meta = None
    begin = []
    end = []
    len_results = []
    results = []
    factors = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    try:
        model_source = sources[request.model_name]
        model_lang = languages[request.model_name]
        model_version = versions[request.model_name]
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.summary_annotator_name,
            version=settings.summary_annotator_version,
            modelName=request.model_name,
            modelVersion=model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.summary_annotator_name} ({settings.summary_annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.summary_annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        mv = ""
        summaries = process_selection(request.model_name, request.all_annotations, request.summary_length)
    except Exception as ex:
        logger.exception(ex)
    return TextImagerResponse(meta=meta, modification_meta=modification_meta, begin=summaries["begin"], end=summaries["end"], summaries=summaries["summary"], model_name=request.model_name, model_version=model_version, model_source=model_source, model_lang=model_lang)



