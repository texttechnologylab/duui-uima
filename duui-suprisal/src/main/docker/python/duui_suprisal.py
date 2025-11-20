from typing import List, Optional
import uvicorn
from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse
from functools import lru_cache
from minicons import scorer
from huggingface_hub import HfApi
from nltk.tokenize import TweetTokenizer
from transformers import AutoConfig
from threading import Lock
import torch
import logging


lru_cache_with_size_model = lru_cache(maxsize=3)
model_load_lock = Lock()

# Documentation response
class DUUIDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    version: str
    # Annotator implementation language (Python, Java, ...)
    implementation_lang: str

class Sentence(BaseModel):
    iBegin: int
    iEnd: int
    sText: str
    sCondition: str
    sTarget: str
    sSuprise: Optional[float] = None


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    selection: List[Sentence]
    model_name: Optional[str]

# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of annotated:
    sentences: List[Sentence]
    model_name: str

class Settings(BaseSettings):
    # Name of the Model
    model_name: Optional[str] = "goldfish-models/spa_latn_1000mb"


# settings + cache
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=3)

config = {"name": "goldfish-models/spa_latn_1000mb"}


# Start fastapi
app = FastAPI(
    docs_url="/api",
    redoc_url=None,
    title="Suprising-Detection",
    description="Suprising-Detection for DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Giuseppe Abrami",
        "url": "https://www.texttechnologylab.org/team/giuseppe-abrami/",
        "email": "abrami@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Load the Lua communication script
communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'typesystem.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["org.texttechnologylab.annotation.neglab.ScorerSentence"],
        "outputs": ["org.texttechnologylab.annotation.neglab.ScorerSentence"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


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
    return communication


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> DUUIDocumentation:

    documentation = DUUIDocumentation(
        annotator_name=settings.duui_tool_name,
        version=settings.duui_tool_version,
        implementation_lang="Python",
    )
    return documentation

word_tokenizer = TweetTokenizer().tokenize

logger = logging.getLogger(__name__)
logger.info("TTLab DUUI Suprisal")

@lru_cache_with_size_model
def load_cache_model(model_name):
    logger.info("Loading model %s", model_name)

    if torch.cuda.is_available():
        print('GPU available')
        m = scorer.IncrementalLMScorer(model_name, 'cuda')
    else:
        print('GPU unavailable')
        m = scorer.IncrementalLMScorer(model_name)

    return m


def load_model(model_name):
    with model_load_lock:
        return load_cache_model(model_name)

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:

    global model
    returnModelName = None

    if request.model_name:
        model = load_model(request.model_name)
        returnModelName = request.model_name
    else:
        model = load_model(settings.model_name)
        returnModelName = settings.model_name

    bBOS = initialize_bos(returnModelName)

    get_word_surprisals(bBOS, request.selection)

    return DUUIResponse(
        sentences = request.selection,
        model_name = returnModelName
    )


#if __name__ == "__main__":
#  uvicorn.run("duui_gte:app", host="0.0.0.0", port=9715, workers=1)


def initialize_bos(model_name: str) -> bool:
    """
    Determines if the model requires a Beginning of Sentence (BOS) token based on its name.

    Args:
        model_name (str): The name of the model.

    Returns:
        bool: True if BOS is required, False otherwise.
    """
    if "gpt2" in model_name or "pythia" in model_name or "SmolLM" in model_name:
        return True
    return False

def get_word_surprisals(BOS, sentences:List[Sentence]):

    for s in sentences:
        get_word_surprisal(BOS, s)

def get_word_surprisal(BOS:bool, sentence:Sentence):

    surprisals = model.word_score_tokenized(
        sentence.sText,
        bos_token=BOS,
        tokenize_function=word_tokenizer,
        surprisal=True,
        bow_correction=True
    )
    result = next((val for word, val in surprisals[0] if word == sentence.sTarget), None)

    sentence.sSuprise = result