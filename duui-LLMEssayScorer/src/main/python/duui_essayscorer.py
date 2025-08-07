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
from EssayScorer import EssayScorer
import torch
model_lock = Lock()

class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int
    typeName: str


class UimaSentenceSelection(BaseModel):
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
    # model_spec_name
    model_spec_name: str
    # # Name of this annotator
    model_version: str
    # #cach_size
    model_cache_size: int
    # # url of the model
    model_source: str
    # # language of the model
    model_lang: str


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemEssayScorer.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_essayscorer.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The texts language
    doc_len: int
    #
    lang: str
    #
    questions:  list
    #
    answers: list



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

def process_selection(questions, answers, model_name: str) -> Dict[str, Union[List[int], List[str], List[float]]]:
    begin = []
    end = []
    results_out = []
    factors = []
    definitions = []
    len_results = []
    # Load the model
    with model_lock:
        model = load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        # Process each question-answer pair
        all_answers = [
            s["text"] for s in answers
        ]
        all_questions = [
            s["text"] for s in questions
        ]
        match model_name:
            case "KevSun/Engessay_grading_ML":
                outputs = model.run_messages(all_answers)
                for i, output in enumerate(outputs):
                    begin.append(answers[i]["begin"])
                    end.append(answers[i]["end"])
                    len_results.append(len(output))
                    results_out.append(all_questions[i])
                    factors.append(output)
                    definitions.append("Essay scoring factors")
            case "JacobLinCool/IELTS_essay_scoring_safetensors":
                outputs = model.run_messages(all_answers)
                for i, output in enumerate(outputs):
                    begin.append(answers[i]["begin"])
                    end.append(answers[i]["end"])
                    len_results.append(len(output))
                    results_out.append(all_questions[i])
                    factors.append(output)
                    definitions.append("Essay scoring factors")
            case _:
                raise ValueError(f"Model {model_name} is not supported.")
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
    # model_name: str
    # model_version: str
    # model_source: str
    # model_lang: str


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="EssayScorer",
    # version=settings.model_version,
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
        processed_sentences = process_selection(request.questions, request.questions, settings.model_name)
        begin.extend(processed_sentences["begin"])
        end.extend(processed_sentences["end"])
        values.extend(processed_sentences["values"])
        keys.extend(processed_sentences["keys"])
        definitions.extend(processed_sentences["definitions"])
        len_results.extend(processed_sentences["len_results"])
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, values=values, keys=keys, definitions=definitions, len_results=len_results)

@lru_cache_with_size
def load_model(model_name: str, device: str) -> EssayScorer:
    """
    Load the model with caching.
    :param model_name: Name of the model to load.
    :param device: Device to load the model on (e.g., 'cpu' or 'cuda').
    :return: Loaded EssayScorer model.
    """
    if model_name in {"KevSun/Engessay_grading_ML", "JacobLinCool/IELTS_essay_scoring_safetensors"}:
        scorer = EssayScorer(model_name=model_name, device=device)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return scorer