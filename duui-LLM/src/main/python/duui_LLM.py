from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from fastapi import FastAPI, Response
from cassis import load_typesystem
from functools import lru_cache
from threading import Lock
import time
from LLMCall import OpenAIProcessing
import json
# from sp_correction import SentenceBestPrediction

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse

model_lock = Lock()




class Settings(BaseSettings):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    annotator_version: str
    # Log level
    log_level: str

# Load settings from env vars
settings = Settings()
# lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# device = 0 if torch.cuda.is_available() else "cpu"
# logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemLLM.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    # logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_LLM.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    text : str
    # The texts language
    lang: str
    #
    len: int
    #
    prompts: Optional[List] = None
    #
    seed: int = None
    #
    temperature: float = None
    #
    url: str
    #
    port: int
    #
    model_name: str


# UIMA type: mark modification of the document
class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# UIMA type: adds metadata to each annotation
class AnnotationMeta(BaseModel):
    name: str
    version: str


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


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
    # begin_prompts
    begin_prompts: List[int]
    end_prompts: List[int]
    id_prompts: List[int]
    responses: List[str]
    contents: List[str]
    additional: List[str]

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="LLM annotator",
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
    modification_timestamp_seconds = int(time.time())
    begin_prompts = []
    end_prompts = []
    responses = []
    contents = []
    additional = []
    id_prompts = []
    try:
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        url = request.url
        port = request.port
        seed = request.seed
        temperature = request.temperature

        text = request.text
        prompts = request.prompts
        model_name = request.model_name
        llm = OpenAIProcessing(url=url, port=port, seed=seed, temperature=temperature)
        # Process the text
        for prompt_i in prompts:
            systemprompt = None if prompt_i["systemPrompt"]["text"]=="" else prompt_i["systemPrompt"]["text"]
            prefix = None if prompt_i["prefix"]["text"]=="" else prompt_i["prefix"]["text"]
            suffix = None if prompt_i["suffix"]["text"]=="" else prompt_i["suffix"]["text"]
            id_prompt = prompt_i["id"]
            prompt_begin = prompt_i["begin"]
            prompt_end = prompt_i["end"]
            begin_prompts.append(prompt_begin)
            end_prompts.append(prompt_end)
            id_prompts.append(id_prompt)
            prompt_text = prompt_i["text"]
            start_time = time.time()
            response_llm = llm.process(prompt_text, model_name, system_prompt=systemprompt, prefix_prompt=prefix, suffix_prompt=suffix)
            time_seconds = time.time() - start_time
            additional.append(json.dumps({"url": url, "port": port, "model_name": request.model_name, "seed": seed, "temperature": temperature, "duration": time_seconds}))
            # logger.debug(f"Processing time: {time_seconds} seconds")
            content = response_llm["choices"][0]["message"]["content"]
            contents.append(content)
            del response_llm["choices"][0]["message"]["content"]
            json_llm_string = json.dumps(response_llm)
            responses.append(json_llm_string)
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, begin_prompts=begin_prompts, end_prompts=end_prompts, id_prompts=id_prompts,responses=responses, contents=contents, additional=additional)
