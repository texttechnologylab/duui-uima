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
from L2SCA_Scorer import l2sca_metrics
import L2SCA_Scorer

model_lock = Lock()
import stanza

NLP_PIPELINES = {
    'en': stanza.Pipeline('en'),
    'de': stanza.Pipeline('de')
}

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
typesystem_filename = 'TypeSystemL2SCA.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_l2sca.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    #
    lang: str
    #
    begin: int
    #
    end: int
    # text
    text: str


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

def process_selection(request):
    begin = request.begin
    end = request.end
    values = []
    measuresList = []
    definitionsList = []
    typeNumberList = []
    typeNameList = []
    codeList = []
    output = {}
    nlp = load_model(request.lang)
    l2sca = l2sca_metrics(request.text, nlp)
    definitions = L2SCA_Scorer.definitions
    measuresL2SCA = L2SCA_Scorer.measures
    typeNumber = L2SCA_Scorer.TypeNumber
    typeName = L2SCA_Scorer.TypeName
    for code_i in l2sca:
        codeList.append(code_i)
        values.append(l2sca[code_i])
        definitionsList.append(definitions[code_i])
        measuresList.append(measuresL2SCA[code_i])
        typeNumberList.append(typeNumber[code_i])
        typeNameList.append(typeName[code_i])
    output["begin"] = begin
    output["end"] = end
    output["values"] = values
    output["codes"] = codeList
    output["definitions"] = definitionsList
    output["measures"] = measuresList
    output["typeNumber"] = typeNumberList
    output["typeName"] = typeNameList
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
    begin: int
    end: int
    values: List
    codes: List
    definitions: List
    measures: List
    typeNumber: List
    typeName: List
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
    begin = 0
    end = 0
    len_results = 0
    group_name = []
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
        output = process_selection(request)
        begin = output["begin"]
        end = output["end"]
        values = output["values"]
        codes = output["codes"]
        definitions = output["definitions"]
        measures = output["measures"]
        typeNumber = output["typeNumber"]
        typeName = output["typeName"]
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, values=values, codes=codes, definitions=definitions, measures=measures, typeNumber=typeNumber, typeName=typeName, model_name=settings.model_name, model_version=model_version, model_source=model_source, model_lang=model_lang)



@lru_cache_with_size
def load_model(lang):
    nlp = NLP_PIPELINES.get(lang)
    return nlp