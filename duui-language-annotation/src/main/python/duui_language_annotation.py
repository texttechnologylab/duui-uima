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
from LanguageDetection import LanguageDetection, LanguageCheck, LanguageIdentification
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse
model_lock = Lock()
sources = {
    "Glotlid": "https://github.com/cisnlp/GlotLID",
    "Fasttext": "https://fasttext.cc/docs/en/language-identification.html",
    "Spacy": "https://github.com/davebulaval/spacy-language-detection",
    "Google": "https://github.com/Mimino666/langdetect",
    "glc3d": "https://github.com/google/cld3",
    "papluca/xlm-roberta-base-language-detection": "https://huggingface.co/papluca/xlm-roberta-base-language-detection",
    "qanastek/51-languages-classifier": "https://huggingface.co/qanastek/51-languages-classifier",
}

# languages = {
#     "02shanky/finetuned-twitter-xlm-roberta-base-emotion": "Multilingual",
#     "DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence": "Multilingual",
#     "pol_emo_mDeBERTa": "Multilingual",
#     "MilaNLProc/xlm-emo-t": "Multilingual",
#     "j-hartmann/emotion-english-distilroberta-base": "en",
#     "michellejieli/emotion_text_classifier": "en",
#     "cardiffnlp/twitter-roberta-base-emotion": "en",
#     "finiteautomata/bertweet-base-emotion-analysis": "en"
# }

versions = {
    "Glotlid": "a9f8a6cf8af1668c09db74bbb427c6255c16bb03",
    "Fasttext": "1142dc4c4ecbc19cc16eee5cdd28472e689267e6",
    "Spacy": "28266a0a15ef5180eb8540bd98ff1c7d14b74e1d",
    "Google": "a1598f1afcbfe9a758cfd06bd688fbc5780177b2",
    "glc3d": "b48dc46512566f5a2d41118c8c1116c4f96dc661",
    "papluca/xlm-roberta-base-language-detection": "9865598389ca9d95637462f743f683b51d75b87b",
    "qanastek/51-languages-classifier": "966ca1a15a30f218ad48561943f046d809d4ed26",
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
    language_annotator_name: str
    # Version of this annotator
    language_annotator_version: str
    # Log level
    language_log_level: str
    # # model_name
    # language_model_name: str
    # Name of this annotator
    language_model_cache_size: int


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.language_model_cache_size)
logging.basicConfig(level=settings.language_log_level)
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
lua_communication_script_filename = "duui_language.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    #
    model_name: str
    #
    selections:  List[UimaSentenceSelection]
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
    lang: List[str]
    scores: List[float]
    model_name: str
    model_version: str
    model_source: str



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.language_annotator_name,
    description="Factuality annotator",
    version=settings.language_annotator_version,
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
        case "Glotlid":
            model_i = LanguageDetection()
        case "Fasttext":
            model_i = LanguageIdentification(model_name="fasttext")
        case "Spacy":
            model_i = LanguageIdentification(model_name="spacy")
        case "Google":
            model_i = LanguageIdentification(model_name="google")
        case "glc3d":
            model_i = LanguageIdentification(model_name="glc3d")
        case _:
            model_i = LanguageCheck(model_name, device=device)
    return model_i


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text

def process_selection(model_name, selection):
    all_begin = []
    all_end = []
    langs = []
    scores = []
    lang_out = {}
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
        results = classifier.lang_prediction(texts)
        for c, res in enumerate(results):
            all_langs = list(res.keys())
            if all_langs[0] not in lang_out:
                lang_out[all_langs[0]] = {}
            begin_i = selection.sentences[c].begin
            end_i = selection.sentences[c].end
            lang_out[all_langs[0]][begin_i] = {
                "begin": begin_i,
                "end": end_i,
                "lang": all_langs[0],
                "score": res[all_langs[0]]
            }
        # sort after begin
        for lang in lang_out:
            lang_out[lang] = dict(sorted(lang_out[lang].items()))
        #fuse begin and end if begin and end+1 of the key are the same
        lang_out_copy = copy.deepcopy(lang_out)
        fused_list = {}
        for lang in lang_out_copy:
            fused_list[lang] = [[]]
            begin_keys = list(lang_out_copy[lang].keys())
            counter_index = 0
            for i in range(len(begin_keys)):
                if i+1 in range(len(begin_keys)):
                    begin_i = lang_out_copy[lang][begin_keys[i+1]]["begin"]
                    end_i = lang_out_copy[lang][begin_keys[i]]["end"]
                    if begin_i == end_i+1:
                        lang_out[lang][begin_keys[i]]["Fuse"] = True
                        lang_out[lang][begin_keys[i+1]]["Fuse"] = True
                        fused_list[lang][counter_index].append(begin_keys[i])
                        fused_list[lang][counter_index].append(begin_keys[i+1])
                    else:
                        if len(fused_list[lang][counter_index]) > 0:
                            counter_index += 1
                            fused_list[lang].append([])
                        if "Fuse" not in lang_out[lang][begin_keys[i]]:
                            fused_list[lang][counter_index].append(begin_keys[i])
                else:
                    if len(fused_list[lang][counter_index]) > 0:
                        counter_index += 1
                        fused_list[lang].append([])
                    if "Fuse" not in lang_out[lang][begin_keys[i]]:
                        fused_list[lang][counter_index].append(begin_keys[i])
        lang_new_out = {}
        for lang in fused_list:
            lang_new_out[lang] = {}
            for i in range(len(fused_list[lang])):
                if len(fused_list[lang][i]) > 0:
                    begin = fused_list[lang][i][0]
                    end = lang_out[lang][fused_list[lang][i][-1]]["end"]
                    scores_i = []
                    for j in range(len(fused_list[lang][i])):
                        scores_i.append(lang_out[lang][fused_list[lang][i][j]]["score"])
                    avg_score = sum(scores_i) / len(scores_i)
                    lang_new_out[lang][begin] = {
                        "begin": begin,
                        "end": end,
                        "lang": lang,
                        "score": avg_score
                    }
        for lang in lang_new_out:
            for begin in lang_new_out[lang]:
                begin = lang_new_out[lang][begin]["begin"]
                end = lang_new_out[lang][begin]["end"]
                langs.append(lang_new_out[lang][begin]["lang"])
                scores.append(lang_new_out[lang][begin]["score"])
                all_begin.append(begin)
                all_end.append(end)
    output = {
        "begin": all_begin,
        "end": all_end,
        "lang": langs,
        "scores": scores
    }
    return output

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):
    # Return data
    meta = None
    begin = []
    end = []
    lang = []
    scores = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    try:
        model_source = sources[request.model_name]
        model_version = versions[request.model_name]
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.language_annotator_name,
            version=settings.language_annotator_version,
            modelName=request.model_name,
            modelVersion=model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.language_annotator_name} ({settings.language_annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.language_annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        mv = ""

        for selection in request.selections:
            processed_sentences = process_selection(request.model_name, selection)
            begin = begin + processed_sentences["begin"]
            end = end + processed_sentences["end"]
            lang = lang + processed_sentences["lang"]
            scores = scores + processed_sentences["scores"]
    except Exception as ex:
        logger.exception(ex)
    return TextImagerResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, lang=lang, scores=scores, model_name=request.model_name,model_version=model_version, model_source=model_source)



