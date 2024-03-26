import copy
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
from Translation import LanguageNLLB, LanguageM2M, WhisperTranslation, TranslationTransformer, language_to_flores200, language_to_MBART, language_match
import json
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse
model_lock = Lock()
sources = {
    "MBART": "https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt",
    "NLLB": "https://huggingface.co/facebook/nllb-200-distilled-600M",
    "Whisper": "https://github.com/openai/whisper",
    "FlanT5Base": "https://huggingface.co/google/flan-t5-base",
}

versions = {
    "MBART": "e30b6cb8eb0d43a0b73cab73c7676b9863223a30",
    "NLLB": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
    "Whisper": "ba3f3cd54b0e5b8ce1ab3de13e32122d0d5f98ab",
    "FlanT5Base": "7bcac572ce56db69c1ea7c8af255c5d7c9672fc2",
}

with open("languages_flores200.json", "r", encoding="UTF-8") as f:
    flores200 = json.load(f)
with open("languages_MBArt.json", "r", encoding="UTF-8") as f:
    mbart_languages = json.load(f)

class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int


class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]

class Settings(BaseSettings):
    # Name of this annotator
    translation_annotator_name: str
    # Version of this annotator
    translation_annotator_version: str
    # Log level
    translation_log_level: str
    # # model_name
    # translation_model_name: str
    # Name of this annotator
    translation_model_cache_size: int


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.translation_model_cache_size)
logging.basicConfig(level=settings.translation_log_level)
logger = logging.getLogger(__name__)

device = 0 if torch.cuda.is_available() else "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemLanguage.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_translation.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    #
    model_name: str
    #
    translation_list: str
    #
    selections: List[dict]


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
    languages: List[str]
    translation: List[str]
    art: List[str]
    model_name: str
    model_version: str
    model_source: str



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.translation_annotator_name,
    description="Factuality annotator",
    version=settings.translation_annotator_version,
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
        case "MBART":
            model_i = LanguageM2M("facebook/mbart-large-50-many-to-many-mmt", device=device)
        case "NLLB":
            model_i = LanguageNLLB("facebook/nllb-200-distilled-600M", device=device)
        case "Whisper":
            model_i = WhisperTranslation(device_i=device, model_name="small")
        case "FlanT5Base":
            model_i = TranslationTransformer(model_name="google/flan-t5-base")
        case _:
            model_i = LanguageNLLB("facebook/nllb-200-distilled-600M", device=device)
    return model_i


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def get_lang(model_name, lang):
    match model_name:
        case "MBART":
            lang_out = language_to_MBART(lang, mbart_languages)
        case "NLLB":
            lang_out = language_to_flores200(lang, flores200)
        case "FlanT5Base":
            lang_out = language_match(lang)
        case _:
            lang_out = lang
    return lang_out


def process_selection(model_name, selection, translations):
    output = {}
    begin = []
    end = []
    text = []
    langin = []
    translation_out = []
    begin_out = []
    end_out = []
    art_out = []
    language_out = []
    translation_list = translations.split(",")
    with model_lock:
        model = load_model(model_name)
        for select_i in selection:
            begin.append(select_i["begin"])
            end.append(select_i["end"])
            text.append(select_i["text"])
            langin.append(select_i["language"])
        for c, text_i in enumerate(text):
            begin_i = begin[c]
            end_i = end[c]
            lang_i = langin[c]
            for c2, lang_out_i in enumerate(translation_list):
                if c2 == 0:
                    lang_input = get_lang(model_name, lang_i)
                    lang_output = get_lang(model_name, lang_out_i)
                    text_in = text_i
                    art_out.append("Normal")
                else:
                    lang_input = get_lang(model_name, translation_list[c2-1])
                    lang_output = get_lang(model_name, lang_out_i)
                    text_in = translation_text
                    art_out.append("Translated")
                end_out.append(end_i)
                translation_text = model.translate(text_in, lang_input[0], lang_output[0])
                translation_out.append(translation_text)
                begin_out.append(begin_i)
                language_out.append(lang_out_i)
    output = {
        "begin": begin_out,
        "end": end_out,
        "translation": translation_out,
        "languages": language_out,
        "art": art_out
    }
    return output

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):
    # Return data
    meta = None
    begin = []
    end = []
    translation = []
    art = []
    languages = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    try:
        model_source = sources[request.model_name]
        model_version = versions[request.model_name]
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.translation_annotator_name,
            version=settings.translation_annotator_version,
            modelName=request.model_name,
            modelVersion=model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.translation_annotator_name} ({settings.translation_annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.translation_annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        processed_sentences = process_selection(request.model_name, request.selections, request.translation_list)
        begin = processed_sentences["begin"]
        end = processed_sentences["end"]
        art = processed_sentences["art"]
        translation = processed_sentences["translation"]
        languages = processed_sentences["languages"]
    except Exception as ex:
        logger.exception(ex)
    return TextImagerResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, languages=languages, translation=translation, art=art, model_name=request.model_name,model_version=model_version, model_source=model_source)



