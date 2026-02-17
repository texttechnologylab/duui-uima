import logging
import io
import gc
import torch
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from cassis import load_typesystem
from fastapi import FastAPI, Response
from starlette.responses import PlainTextResponse
from fastapi.responses import JSONResponse
from typing import List, Optional
from models.duui_models import (
    DUUIMMRequest,
    DUUIMMResponse,
    MultiModelModes,
    ImageType,
    AudioType,
    VideoTypes,
    LLMResult,
    Settings
)

from models.ollama_models import OllamaConfig, OllamaRequest, OllamaResponse
from services.ollama_client import OllamaClient
from services.utils import encode_file_to_base64, map_duui_to_ollama, convert_base64_to_image, convert_base64_to_audio

from fastapi.encoders import jsonable_encoder

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


# Global cache
_loaded_models = {}
_loaded_processors = {}


# Load settings from env vars
settings = Settings()

lua_communication_script, logger, type_system, device = None, None, None, None

def init():
    global lua_communication_script, logger, type_system, device


    logging.basicConfig(level=settings.log_level)
    logger = logging.getLogger(__name__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    logger.info(f'USING {device}')
    # Load the predefined typesystem that is needed for this annotator to work
    # typesystem_filename = './TypeSystemMM.xml'
    typesystem_filename = '../resources/TypeSystemMM.xml'
    # logger.debug("Loading typesystem from \"%s\"", typesystem_filename)


    logger.debug("*"*20 + "Lua communication script" + "*"*20)
        # Load the Lua communication script
    lua_communication_script_filename = "duui_webUIWrapper.lua"


    with open(lua_communication_script_filename, 'rb') as f:
        lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

    with open(typesystem_filename, 'rb') as f:
        type_system = load_typesystem(f)


init()

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="Wrapper for Ollama/OpenWebUI API with DUUI compatibility",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Ali Abusaleh, TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "a.abusaleh@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    # TODO rimgve cassis dependency, as only needed for typesystem at the moment?
    xml = type_system.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["string", "org.texttechnologylab.annotation.type.Image"],
        "outputs": ["string", "org.texttechnologylab.annotation.type.Image"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)

# Return documentation info
@app.get("/v1/documentation")
def get_documentation():
    return JSONResponse({"documents": "testing"})
    # return DUUIMMDocumentation(
    #     annotator_name=settings.image_to_text_annotator_name,
    #     version=settings.image_to_text_model_version,
    #     implementation_lang="Python",
    #     meta={
    #         "log_level": settings.log_level,
    #         "model_version": settings.model_version,
    #         "model_cache_size": settings.model_cache_size,
    #         # "models": sources,
    #         # "languages": languages,
    #         # "versions": versions,
    #     },
    #     parameters={
    #         "prompt": "Prompt",
    #         "doc_lang": "Document language",
    #         "model_name": "Model name",
    #         "individual": "A flag for processing the images as one (set of frames) or indivisual. Note: it only works in a complex-mode",
    #         "mode": "a mode of operation"
    #
    #     }
    # )

# --- API Endpoints ---
@app.post("/v1/process", response_model=DUUIMMResponse)
async def process_ollama(duui_request: DUUIMMRequest):
    # Parse the JSON request body
    # request_data = request.json()
    # duui_request = request.json().load()

    # Initialize Ollama client
    config = OllamaConfig()
    config.host = duui_request.ollama_host
    config.port = duui_request.ollama_port
    config.auth_token = duui_request.ollama_auth_token
    client = OllamaClient(config)

    # Encode files (if present in the request)
    # encoded_images = [encode_file_to_base64(convert_base64_to_image(img.src)) for img in duui_request.images] if duui_request.images else None
    encoded_images = [img.src for img in duui_request.images] if duui_request.images else None
    encoded_audios = [encode_file_to_base64(convert_base64_to_audio(aud)) for aud in duui_request.audios] if duui_request.audios else None
    encoded_videos = [encode_file_to_base64(vid) for vid in duui_request.videos] if duui_request.videos else None

    # TODO I need to support this.
    system_prompt = duui_request.system_prompt

    Responses = []
    Errors = []
    # iterate over duui_request.prompts and make a Request per it.
    for prompt in duui_request.prompts:
        # Map DUUI request to Ollama request
        ollama_request = map_duui_to_ollama(duui_request.model_name, system_prompt, prompt, encoded_images, encoded_audios, encoded_videos)
        # Call Ollama
        ollama_response = client.generate(ollama_request)

        if ollama_response.response:
            Responses.append(LLMResult(meta=ollama_response.response, prompt_ref=0, message_ref="0"))

        if ollama_response.error:
            Errors.append(LLMResult(meta=ollama_response.error, prompt_ref=0, message_ref="0"))

    # Map Ollama response to DUUIMMResponse
    return DUUIMMResponse(
        processed_text=Responses if Responses else None,
        model_name=duui_request.model_name,
        model_source="Ollama/OpenWebUI",
        model_lang=duui_request.doc_lang,
        model_version="1.0.0",
        errors=Errors if Errors else None,
        prompts=duui_request.prompts,
    )

# --- Health Check ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("duui_webUIWrapper:app", host="0.0.0.0", port=9714, workers=1)

