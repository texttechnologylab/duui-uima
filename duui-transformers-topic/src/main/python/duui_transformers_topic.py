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
from TopicSpeech import TopicCheck, TopicCheckSetFit
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse

model_lock = Lock()
sources = {
    "KnutJaegersberg/topic-classification-IPTC-subject-labels": "https://huggingface.co/KnutJaegersberg/topic-classification-IPTC-subject-labels",
    "poltextlab/xlm-roberta-large-manifesto-cap": "https://huggingface.co/poltextlab/xlm-roberta-large-manifesto-cap",
    "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1": "https://huggingface.co/manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1",
    "cardiffnlp/tweet-topic-latest-single": "https://huggingface.co/cardiffnlp/tweet-topic-latest-single",
    "chkla/parlbert-topic-german": "https://huggingface.co/chkla/parlbert-topic-german",
    "classla/xlm-roberta-base-multilingual-text-genre-classifier": "https://huggingface.co/classla/xlm-roberta-base-multilingual-text-genre-classifier",
    "ssharoff/genres": "https://huggingface.co/ssharoff/genres",
}

languages = {
    "KnutJaegersberg/topic-classification-IPTC-subject-labels": "Multi",
    "poltextlab/xlm-roberta-large-manifesto-cap": "Multi",
    "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1": "Multi",
    "cardiffnlp/tweet-topic-latest-single": "EN",
    "chkla/parlbert-topic-german": "DE",
    "classla/xlm-roberta-base-multilingual-text-genre-classifier": "Multi",
    "ssharoff/genres": "EN",
}

versions = {
    "KnutJaegersberg/topic-classification-IPTC-subject-labels": "fe1fb726c12850b1e2f6ed3fa379a0a6c4558a4c",
    "poltextlab/xlm-roberta-large-manifesto-cap": "5f19b49c412d504c1c8357a31367a65c0302717e",
    "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1": "06c046795a3b7b9822755f0a73776f8fabec3977",
    "cardiffnlp/tweet-topic-latest-single": "0ff86a9d19a5bb4045dd7ebced3714796890cfbe",
    "chkla/parlbert-topic-german": "df343699abeb22e08c096ab3974cfd35877ce47f",
    "classla/xlm-roberta-base-multilingual-text-genre-classifier": "de7ed0ff1063e1e4bd3fd1bdda54e3ad85fb5419",
    "ssharoff/genres": "dc9cb7ef031abc96081d9ea96aa0e2ee1636ce04",
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
    # # model_name
    # model_name: str
    # Name of this annotator
    model_version: str
    # cach_size
    model_cache_size: int


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemTopic.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_transformers_topic.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The texts language
    doc_len: int
    #
    lang: str
    #
    model_name: str
    #
    selections: List[UimaSentenceSelection]
    #


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
class DUUIResponse(BaseModel):
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
    results: List
    factors: List
    len_results: List[int]
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
def load_model(model_name):
    if model_name=="KnutJaegersberg/topic-classification-IPTC-subject-labels":
        model_i = TopicCheckSetFit(model_name)
    else:
        model_i = TopicCheck(model_name, device)
    return model_i


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def process_selection(model_name, selection):
    begin = []
    end = []
    results_out = []
    factors = []
    len_results = []
    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    texts = [
        s.text
        for s in selection.sentences
    ]
    logger.debug("Preprocessed texts:")
    logger.debug(texts)

    with model_lock:
        classifier = load_model(model_name)

        results = classifier.topic_prediction(texts)
        for c, res in enumerate(results):
            res_i = []
            factor_i = []
            sentence_i = selection.sentences[c]
            begin_i = sentence_i.begin
            end_i = sentence_i.end
            len_rel = len(res)
            begin.append(begin_i)
            end.append(end_i)
            for i in res:
                res_i.append(i)
                factor_i.append(res[i])
            len_results.append(len_rel)
            results_out.append(res_i)
            factors.append(factor_i)
    output = {
        "begin": begin,
        "end": end,
        "len_results": len_results,
        "results": results_out,
        "factors": factors
    }

    return output, versions[model_name]


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest):
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
        lang_document = request.lang
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=request.model_name,
            modelVersion=model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        mv = ""

        for selection in request.selections:
            processed_sentences, model_version_2 = process_selection(request.model_name, selection)
            begin = begin + processed_sentences["begin"]
            end = end + processed_sentences["end"]
            len_results = len_results + processed_sentences["len_results"]
            results = results + processed_sentences["results"]
            factors = factors + processed_sentences["factors"]
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, results=results,
                        len_results=len_results, factors=factors, model_name=request.model_name,
                        model_version=model_version, model_source=model_source, model_lang=model_lang)