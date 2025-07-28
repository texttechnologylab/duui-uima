from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
from functools import lru_cache
from threading import Lock
from starlette.responses import PlainTextResponse
from LLMDetection import RoBERTaClassifier, Radar, Binoculars, E5Lora, DetectLLM_LRR, FastDetectGPT, FastDetectGPTwithScoring, MachineTextDetector, AIGCDetector, SuperAnnotate, FakeSpotAI, Desklib, Mage, HC3AIDetect, ArguGPT, DetectAIve, AIDetectModel, LogRank, Wild, T5Sentinel, OpenAIDetector, PirateXXAIDetector
from PHDScore import  PDHScorer
import torch
model_lock = Lock()

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
    # model_name
    model_name: str
    # Name of this annotator
    model_version: str
    #cach_size
    model_cache_size: int
    # url of the model
    model_source: str
    # language of the model
    model_lang: str


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemLLMDetection.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_llmdetection.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The texts language
    doc_len: int
    #
    lang: str
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
    definitions = []
    len_results = []
    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    texts = [
        s.text
        for s in selection.sentences
    ]
    with model_lock:
        classifier = load_model(model_name)

        results = classifier.process_texts(texts)
        for c, res in enumerate(results):
            res_i = []
            factor_i = []
            sentence_i = selection.sentences[c]
            begin_i = sentence_i.begin
            end_i = sentence_i.end
            len_rel = len(res)
            begin.append(begin_i)
            end.append(end_i)
            def_i = []
            for i in res:
                res_i.append(i)
                factor_i.append(res[i])
                if i == "Human":
                    def_i.append("Probability that the sentence is human written")
                elif i == "LLM":
                    def_i.append("Probability that the sentence is written by a large language model (LLM)")
                elif i == "Binocular-Score":
                    def_i.append("Binoculars Score for the sentence")
                elif i == "DetectLLM-LRR":
                    def_i.append("DetectLLM Log-Likelihood Log-Rank Ratio Score for the sentence")
                elif i == "Fast-DetectGPT":
                    def_i.append("Fast-DetectGPT Score for the sentence (same model for both reference and scoring)")
                elif i == "Fast-DetectGPTwithScoring":
                    def_i.append("Fast-DetectGPTwithScoring Score for the sentence (different model for reference and scoring)")
                elif i == "PHD":
                    def_i.append("PHD Score for the sentence (Intrinsic Dimensionality)")
                elif i == "MLE":
                    def_i.append("MLE Score for the sentence (Intrinsic Dimensionality)")
                elif i == "Machine-Polished":
                    def_i.append("Probability that the sentence is machine polished")
                elif i == "Machine-Humanized":
                    def_i.append("Probability that the sentence is machine humanized")
                elif i == "LogRank":
                    def_i.append("Log-Rank Score for the sentence")
                else :
                    def_i.append("Other Metric")
            definitions.append(def_i)
            len_results.append(len_rel)
            results_out.append(res_i)
            factors.append(factor_i)
    output = {
        "begin": begin,
        "end": end,
        "len_results": len_results,
        "keys": results_out,
        "values": factors,
        "definitions": definitions,
    }
    return output



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
    values: List
    keys: List
    definitions: List
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
    description="Offensive annotator",
    version=settings.model_version,
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


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest):
    # Return data
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    begin = []
    end = []
    values = []
    keys = []
    definitions = []
    len_results = []
    try:
        model_source = settings.model_source
        model_lang = settings.model_lang
        model_version = settings.model_version
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=settings.model_name,
            modelVersion=settings.model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        for selection in request.selections:
            processed_sentences = process_selection(settings.model_name, selection)
            begin.extend(processed_sentences["begin"])
            end.extend(processed_sentences["end"])
            values.extend(processed_sentences["values"])
            keys.extend(processed_sentences["keys"])
            definitions.extend(processed_sentences["definitions"])
            len_results.extend(processed_sentences["len_results"])
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, values=values, keys=keys, definitions=definitions, len_results=len_results,model_name=settings.model_name, model_version=settings.model_version, model_source=settings.model_source, model_lang=settings.model_lang)



@lru_cache_with_size
def load_model(model_name: str) -> Union[RoBERTaClassifier, Radar, Binoculars]:
    match model_name:
        case "HelloSimpleAI":
            model_i = RoBERTaClassifier()
        case "Radar":
            model_i = Radar()
        case "Binocular":
            model_i = Binoculars()
        case "E5LoRA":
            model_i = E5Lora()
        case "DetectLLM-LRR":
            model_i = DetectLLM_LRR("gpt2")
        case "Fast-DetectGPT":
            model_i = FastDetectGPT("gpt2")
        case "Fast-DetectGPTwithScoring":
            model_i = FastDetectGPTwithScoring("EleutherAI/gpt-neo-1.3B","EleutherAI/gpt-neo-125m")
        case "MachineTextDetector":
            model_i = MachineTextDetector()
        case "AIGCDetectorEn":
            model_i = AIGCDetector("en")
        case "AIGCDetectorZh":
            model_i = AIGCDetector("zh")
        case "SuperAnnotate":
            model_i = SuperAnnotate()
        case "FakeSpotAI":
            model_i = FakeSpotAI()
        case "Desklib":
            model_i = Desklib()
        case "Mage":
            model_i = Mage()
        case "PHDScore":
            model_i = PDHScorer("FacebookAI/xlm-roberta-base", "cuda" if torch.cuda.is_available() else "cpu", alpha=1.0, metric="euclidean", n_points=9, n_reruns=3)
        case "HC3AIDetect":
            model_i = HC3AIDetect()
        case "ArguGPTSentence":
            model_i = ArguGPT()
        case "ArguGPTDocument":
            model_i = ArguGPT("SJTU-CL/RoBERTa-large-ArguGPT")
        case "DetectAIve":
            model_i = DetectAIve()
        case "AIDetectModel":
            model_i = AIDetectModel()
        case "LogRank":
            model_i = LogRank()
        case "Wild":
            model_i = Wild()
        case "T5Sentinel":
            model_i = T5Sentinel()
        case "OpenAIDetector":
            model_i = OpenAIDetector()
        case "PirateXXAIDetector":
            model_i = PirateXXAIDetector()
        case _:
            print("ModelName")
            raise ValueError(f";{model_name};")
    return model_i

