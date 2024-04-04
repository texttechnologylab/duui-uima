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
from StanceDetection import TransformerStance, zeroshotClassification, ChatGPT
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse

model_lock = Lock()
# sources = {
#     "CHKLA": "https://huggingface.co/chkla/roberta-argument",
#     "UKP": "https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering",
#     "UKPLARGE": "https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering",
#     "Gpt4": "https://github.com/openai/openai-cookbook",
#     "Gpt3.5": "https://github.com/openai/openai-cookbook",
# }
#
# versions = {
#     "CHKLA": "7c0e6b88c91828ba07dfc473d2d11628e3b734fc",
#     "UKP": "72f643b06a06b9ba82a25df2c134664fc26f84f3",
#     "UKPLARGE": "72f643b06a06b9ba82a25df2c134664fc26f84f3",
#     "Gpt4": "63336788349e400fdbcf08c66e98b1e5b5209736",
#     "Gpt3.5": "63336788349e400fdbcf08c66e98b1e5b5209736",
# }
#
# languages = {
#     "CHKLA": "en",
#     "UKP": "en",
#     "UKPLARGE": "en",
#     "Gpt4": "Multilingual",
#     "Gpt3.5": "Multilingual",
# }
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
    annotator_name: str
    # Version of this annotator
    annotator_version: str
    # Log level
    log_level: str
    model_name: str
    model_version: str
    model_url: str
    model_lang: str
    # model_name: str
    # Name of this annotator
    model_cache_size: int
    chatgpt_key: str


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

device = 0 if torch.cuda.is_available() else "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemStance.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_stance.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    hypothesis: List[dict]
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
    begin_stances: List[int]
    end_stances: List[int]
    predictions: List[Dict[str, Union[str, float]]]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="Factuality annotator",
    version=settings.annotator_version,
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
        case "mlburnham":
            model_i = zeroshotClassification("mlburnham/deberta-v3-base-polistance-affect-v1.0", device)
        case "kornosk":
            model_i = TransformerStance("kornosk/bert-election2020-twitter-stance-trump", device)
        case "gpt4":
            model_i = ChatGPT("gpt-4", chatgpt_key)
        case "gpt3.5":
            model_i = ChatGPT("gpt-3.5-turbo", chatgpt_key)
        case _:
            settings.model_name = "mlburnham"
            model_i = zeroshotClassification("mlburnham/deberta-v3-base-polistance-affect-v1.0", device)
    return model_i


model = load_model(settings.model_name, settings.chatgpt_key)


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


dict_names = {
    "is neutral towards": "neutral",
    "supports": "support",
    "opposes": "oppose",
    "AGAINST": "oppose",
    "FAVOR": "support",
    "NONE": "neutral",
    0: "neutral",
    1: "support",
    2: "oppose"
}


def process_selection(model, hypothesis):
    input = {}
    output = {
        "begin": [],
        "end": [],
        "begin_stances": [],
        "end_stances": [],
        "predictions": [],
    }
    for i, hyp in enumerate(hypothesis):
        begin = hyp["begin"]
        end = hyp["end"]
        text = hyp["text"]
        stances = hyp["stances"]
        stances_list = []
        begin_stances = []
        end_stances = []
        for stance in stances:
            stances_list.append(stance["text"])
            begin_stances.append(stance["begin"])
            end_stances.append(stance["end"])
        input[f"{begin}_{end}"] = {"text": text, "begin": begin, "end": end, "stances": stances_list,
                                   "begin_stances": begin_stances, "end_stances": end_stances}
        prediction = model.predict(stances_list, text)
        output_prediction = []
        for c, pred in enumerate(prediction):
            dict_i = {}
            if settings.model_name == "gpt4" or settings.model_name == "gpt3.5":
                label = pred["label"]
                dict_i["label"] = dict_names[label]
                if "confidence" in pred:
                    dict_i["confidence"] = pred["confidence"]
                else:
                    dict_i["confidence"] = 1
                if "reason" in pred:
                    dict_i["reason"] = pred["reason"]
                else:
                    dict_i["reason"] = ""
            else:
                for key, value in pred.items():
                    dict_i[dict_names[key]] = value
            output_prediction.append(dict_i)
            output["begin"].append(begin)
            output["end"].append(end)
            output["begin_stances"].append(begin_stances[c])
            output["end_stances"].append(end_stances[c])
            output["predictions"].append(dict_i)
    return output


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):
    # Return data
    meta = None
    begin = []
    end = []
    begin_stances = []
    end_stances = []
    predictions = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    chatgpt_key = ""
    if isinstance(chatgpt_key, str):
        chatgpt_key = request.chatgpt_key
    try:
        model_source = settings.model_url
        model_version = settings.model_version
        model_lang = settings.model_lang
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=settings.model_name,
            modelVersion=model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        if settings.model_name == "gpt4" or settings.model_name == "gpt3.5":
            model_i = load_model(settings.model_name, chatgpt_key)
            processed_sentences = process_selection(model_i, request.hypothesis)
        else:
            processed_sentences = process_selection(model, request.hypothesis)
        begin = processed_sentences["begin"]
        end = processed_sentences["end"]
        begin_stances = processed_sentences["begin_stances"]
        end_stances = processed_sentences["end_stances"]
        predictions = processed_sentences["predictions"]
    except Exception as ex:
        logger.exception(ex)
    return TextImagerResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end,
                              model_name=settings.model_name,
                              begin_stances=begin_stances, end_stances=end_stances, predictions=predictions,
                              model_version=model_version, model_source=model_source, model_lang=model_lang)
