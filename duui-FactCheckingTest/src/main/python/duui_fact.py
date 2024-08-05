from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
import torch
from functools import lru_cache
from factchecker import UniEvalFactCheck, NubiaFactCheck
from threading import Lock
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse

model_lock = Lock()

sources = {
    "nubia": "https://github.com/wl-research/nubia",
    "unieval": "https://github.com/maszhongming/UniEval"
}

languages = {
    "nubia": "en",
    "unieval": "en"
}


class Settings(BaseSettings):
    # Name of this annotator
    fact_annotator_name: str
    # Version of this annotator
    fact_annotator_version: str
    # Log level
    fact_log_level: str
    # model_name
    fact_model_name: str
    # Name of this annotator
    fact_model_version: str
    # cach_size
    fact_model_cache_size: int


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.fact_model_cache_size)
logging.basicConfig(level=settings.fact_log_level)
logger = logging.getLogger(__name__)

device = 0 if torch.cuda.is_available() else "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemFactChecking.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_fact.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    claims_all: Optional[list] = None
    #
    facts_all: Optional[list] = None
    # Optional map/dict of parameters
    # TODO how with lua?
    parameters: Optional[dict] = None


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
    begin_claims: List[int]
    end_claims: List[int]
    begin_facts: List[int]
    end_facts: List[int]
    consistency: List[float]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.fact_annotator_name,
    description="Factuality annotator",
    version=settings.fact_model_version,
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
    meta = None
    checkfacts = {}
    consistency = []
    begin_claims = []
    end_claims = []
    begin_facts = []
    end_facts = []
    # Save modification start time for later
    modification_timestamp_seconds = int(time())
    try:
        model_source = sources[settings.fact_model_name]
        model_lang = languages[settings.fact_model_name]
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.fact_annotator_name,
            version=settings.fact_annotator_version,
            modelName=settings.fact_model_name,
            modelVersion=settings.fact_model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.fact_annotator_name} ({settings.fact_annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.fact_annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        claims = request.claims_all
        facts = request.facts_all
        with model_lock:
            model_run = load_model(settings.fact_model_name)
            claim_list = []
            fact_list = []
            counters = []
            for c, claim in enumerate(claims):
                for fc, fact_i in enumerate(claim["facts"]):
                    claim_list.append(claim["text"])
                    fact_list.append(fact_i["text"])
                    counters.append(f"{c}_{fc}")
            factchecked = model_run.check(claim_list, fact_list)
            for c, fact_check_i in enumerate(factchecked):
                checkfacts[counters[c]] = fact_check_i
            # check fact_list
            claim_list = []
            fact_list = []
            counters = []
            factscheck = {}
            for fc, fact_i in enumerate(facts):
                for c, claim in enumerate(fact_i["claims"]):
                    if f"{c}_{fc}" not in checkfacts:
                        claim_list.append(claim["text"])
                        fact_list.append(fact_i["text"])
                        counters.append(f"{c}_{fc}")
            if len(claim_list) > 0:
                factchecked = model_run.check(claim_list, fact_list)
            else:
                factchecked = {}
            for c, fact_check_i in enumerate(factchecked):
                factscheck[counters[c]] = fact_check_i
            for key_i in checkfacts:
                key_claim = int(key_i.split("_")[0])
                key_facts = int(key_i.split("_")[1])
                cons = checkfacts[key_i]['consistency']
                consistency.append(cons)
                claim_i = claims[key_claim]
                begin_claims.append(claim_i["begin"])
                end_claims.append(claim_i["end"])
                fact_i = claim_i["facts"][key_facts]
                begin_facts.append(fact_i["begin"])
                end_facts.append(fact_i["end"])
            for key_i in factscheck:
                key_claim = int(key_i.split("_")[0])
                key_facts = int(key_i.split("_")[1])
                cons = checkfacts[key_i]['consistency']
                consistency.append(cons)
                fact_i = facts[key_facts]
                claim_i = fact_i["claims"][key_claim]
                begin_claims.append(claim_i["begin"])
                end_claims.append(claim_i["end"])
                begin_facts.append(fact_i["begin"])
                end_facts.append(fact_i["end"])
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, consistency=consistency,
                        begin_claims=begin_claims, end_claims=end_claims, begin_facts=begin_facts, end_facts=end_facts,
                        model_name=settings.fact_model_name, model_version=settings.fact_model_version,
                        model_source=model_source, model_lang=model_lang)


@lru_cache_with_size
def load_model(model_name):
    if model_name == "nubia":
        model_i = NubiaFactCheck()
    else:
        model_i = UniEvalFactCheck(device=device)
    return model_i
