import copy
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union, Any
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
import torch
from threading import Lock
from functools import lru_cache
import json
from ArgumentClassification import TransformerArgument, UkpArgument, ChatGPT
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse

model_lock = Lock()
sources = {
    "CHKLA": "https://huggingface.co/chkla/roberta-argument",
    "UKP": "https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering",
    "UKPLARGE": "https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering",
    "Gpt4": "https://github.com/openai/openai-cookbook",
    "Gpt3.5": "https://github.com/openai/openai-cookbook",
}

versions = {
    "CHKLA": "7c0e6b88c91828ba07dfc473d2d11628e3b734fc",
    "UKP": "72f643b06a06b9ba82a25df2c134664fc26f84f3",
    "UKPLARGE": "72f643b06a06b9ba82a25df2c134664fc26f84f3",
    "Gpt4": "63336788349e400fdbcf08c66e98b1e5b5209736",
    "Gpt3.5": "63336788349e400fdbcf08c66e98b1e5b5209736",
}

languages = {
    "CHKLA": "en",
    "UKP": "en",
    "UKPLARGE": "en",
    "Gpt4": "Multilingual",
    "Gpt3.5": "Multilingual",
}
labels_to_name = {
    0: "Argument_for",
    1: "Argument_against",
    2: "NoArgument"
}


class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int


class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]


class Settings(BaseSettings):
    # Name of this annotator
    argument_annotator_name: str
    # Version of this annotator
    argument_annotator_version: str
    # Log level
    argument_log_level: str
    # # model_name
    # argument_model_name: str
    # Name of this annotator
    argument_model_cache_size: int
    chatgpt_key: str


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.argument_model_cache_size)
logging.basicConfig(level=settings.argument_log_level)
logger = logging.getLogger(__name__)

device = 0 if torch.cuda.is_available() else "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemArgument.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_argument.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    #
    model_name: str
    #
    selections: List[dict]
    #
    topic: str
    #
    chatgpt_key: Optional[Any]


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
    keys: List[List[str]]
    values: List[List[Union[str, float]]]
    length: List[int]
    topics: List[str]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.argument_annotator_name,
    description="Factuality annotator",
    version=settings.argument_annotator_version,
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
def load_model(model_name, chatgpt_key=""):
    match model_name:
        case "CHKLA":
            model_i = TransformerArgument("chkla/roberta-argument", device)
        case "UKP":
            model_i = UkpArgument("models/argument_classification_ukp_all_data", device)
        case "UKPLARGE":
            model_i = UkpArgument("models/argument_classification_ukp_all_data_large_model", device)
        case "Gpt4":
            model_i = ChatGPT("gpt-4", chatgpt_key)
        case "Gpt3.5":
            model_i = ChatGPT("gpt-3.5-turbo", chatgpt_key)
        case _:
            model_i = UkpArgument("models/argument_classification_ukp_all_data", device)
    return model_i


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def process_selection(model_name, selections, chatgpt_key, topic):
    begin = []
    end = []
    text = []
    for selection in selections:
        text.append(selection["text"])
        end.append(selection["end"])
        begin.append(selection["begin"])
    with model_lock:
        model = load_model(model_name, chatgpt_key)
        cache_size = 10
        splitted_text = [text[i:i + cache_size] for i in range(0, len(text), cache_size)]
        keys = []
        values = []
        len_tables = []
        topics = []
        begin_out = []
        ends_out = []
        counter = 0
        for split_i in splitted_text:
            out_present = model.predict(split_i, topic)
            for out_i in out_present:
                if "error" in out_i:
                    if out_i["error"]:
                        counter = + 1
                        continue
                if model_name == "Gpt3.5" or model_name == "Gpt4":
                    list_key = []
                    list_value = []
                    if "label" in out_i:
                        list_value.append(labels_to_name[out_i["label"]])
                        list_key.append("label")
                    else:
                        counter = + 1
                        continue
                    if "confidence" in out_i:
                        con = int(out_i["confidence"])
                        list_value.append(float(con*0.01))
                        list_key.append("confidence")
                    if "reason" in out_i:
                        list_value.append(out_i["reason"])
                        list_key.append("reason")
                    keys.append(list_key)
                    values.append(list_value)
                    len_tables.append(len(list_key))
                    topics.append(topic)
                    begin_out.append(begin[counter])
                    ends_out.append(end[counter])
                else:
                    keys.append(list(out_i.keys()))
                    values.append(list(out_i.values()))
                    len_tables.append(len(out_i))
                    topics.append(topic)
                    begin_out.append(begin[counter])
                    ends_out.append(end[counter])
                counter = + 1
        output = {
            "begin": begin_out,
            "end": ends_out,
            "keys": keys,
            "values": values,
            "length": len_tables,
            "topics": topics
        }
    return output


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):
    # Return data
    meta = None
    begin = []
    end = []
    keys = []
    values = []
    length = []
    topics = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    chatgpt_key = ""
    if isinstance(chatgpt_key, str):
        chatgpt_key = request.chatgpt_key
    try:
        model_source = sources[request.model_name]
        model_version = versions[request.model_name]
        model_lang = languages[request.model_name]
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.argument_annotator_name,
            version=settings.argument_annotator_version,
            modelName=request.model_name,
            modelVersion=model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.argument_annotator_name} ({settings.argument_annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.argument_annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        processed_sentences = process_selection(request.model_name, request.selections, chatgpt_key, request.topic)
        begin = processed_sentences["begin"]
        end = processed_sentences["end"]
        keys = processed_sentences["keys"]
        values = processed_sentences["values"]
        length = processed_sentences["length"]
        topics = processed_sentences["topics"]
    except Exception as ex:
        logger.exception(ex)
    return TextImagerResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, keys=keys,
                              values=values, length=length, topics=topics, model_name=request.model_name,
                              model_version=model_version, model_source=model_source, model_lang=model_lang)
