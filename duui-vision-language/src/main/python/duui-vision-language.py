import base64
import logging
from functools import lru_cache
from http.client import responses
from threading import Lock
import gc
import torch
import uvicorn
from PIL import Image
from cassis import load_typesystem
from fastapi import FastAPI, Response
from starlette.responses import PlainTextResponse, JSONResponse

from models.duui_api_models import DUUIMMRequest, DUUIMMResponse, ImageType, Settings, DUUIMMDocumentation, MultiModelModes, LLMResult, LLMPrompt
from models.Molmo import MolmoE1BModel, Molmo7BOModel, Molmo7BDModel, Molmo72BModel, Molmo7BDModelVLLM

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Cache
_loaded_models = {}
_loaded_processors = {}

settings = Settings()
lru_cache_with_size = lru_cache(maxsize=int(settings.mm_model_cache_size))

lua_communication_script, logger, type_system, device = None, None, None, None

def init():
    global lua_communication_script, logger, type_system, device

    logging.basicConfig(level=settings.mm_log_level)
    logger = logging.getLogger(__name__)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"USING {device}")

    with open("duui-vision-language.lua", 'rb') as f:
        lua_communication_script = f.read().decode("utf-8")
    with open("../resources/TypeSystemVisionLanguage.xml", 'rb') as f:
        type_system = load_typesystem(f)

sources = {
    "Molmo-72B-0924": "https://huggingface.co/allenai/Molmo-72B-0924",
    "Molmo-7B-D-0924": "https://huggingface.co/allenai/Molmo-7B-D-0924",
    "Molmo-7B-O-0924": "https://huggingface.co/allenai/Molmo-7B-O-0924",
    "MolmoE-1B-0924": "https://huggingface.co/allenai/MolmoE-1B-0924",
    "Molmo7BDModelVLLM": "https://huggingface.co/allenai/Molmo-7B-D-0924"
}
languages = {
    "Molmo-72B-0924": "en",
    "Molmo-7B-D-0924": "en",
    "Molmo-7B-O-0924": "en",
    "MolmoE-1B-0924": "en",
    "Molmo7BDModelVLLM": "en",
}
versions = {
    "Molmo-72B-0924": "2ca845922396b7a5f7086bfda3fca6b8ecd1c8f3",
    "Molmo-7B-D-0924": "ac032b93b84a7f10c9578ec59f9f20ee9a8990a2",
    "Molmo-7B-O-0924": "0e727957abd46f3ef741ddbda3452db1df873a6e",
    "MolmoE-1B-0924": "69e3445d130507eadaa9123e3c411ce17aeb8afa",
    "Molmo7BDModelVLLM": "ac032b93b84a7f10c9578ec59f9f20ee9a8990a2",
}

model_lock = Lock()
init()

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.mm_annotator_name,
    description="DUUI component for Molmo multimodal",
    version=settings.mm_annotator_version
)

@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=type_system.to_xml().encode("utf-8"),
        media_type="application/xml"
    )

@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script

@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    return JSONResponse(content={
        "inputs": ["string", "org.texttechnologylab.annotation.type.Image"],
        "outputs": ["string", "org.texttechnologylab.annotation.type.Image"]
    })

@app.get("/v1/documentation")
def get_documentation():
    return DUUIMMDocumentation(
        annotator_name=settings.image_to_text_annotator_name,
        version=settings.image_to_text_model_version,
        implementation_lang="Python",
        meta={
            "log_level": settings.mm_log_level,
            "model_version": settings.mm_model_version,
            "model_cache_size": settings.mm_model_cache_size,
            "models": sources,
            "languages": languages,
            "versions": versions
        },
        parameters={
            "prompt": "Prompt",
            "doc_lang": "Document language",
            "model_name": "Model name",
            "individual": "Treat each input individually",
            "mode": "Operation mode"
        }
    )

@lru_cache_with_size
def load_model(model_name, device=device):
    if model_name == "MolmoE1BModel":
        return MolmoE1BModel(device=device, revision=versions.get(model_name))
    elif model_name == "Molmo7BOModel":
        return Molmo7BOModel(device=device, revision=versions.get(model_name))
    elif model_name == "Molmo7BDModel":
        return Molmo7BDModel(device=device, revision=versions.get(model_name))
    elif model_name == "Molmo72BModel":
        return Molmo72BModel(device=device, revision=versions.get(model_name))
    elif model_name == "Molmo7BDModelVLLM":
        return Molmo7BDModelVLLM()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def process_text_only(model_name: str, prompt: LLMPrompt) -> LLMResult:
    return load_model(model_name).process_text(prompt)

def process_image_only(model_name: str, image_base64: str, prompt: LLMPrompt) -> LLMResult:
    return load_model(model_name).process_image(image_base64, prompt)

@app.post("/v1/process")
def post_process(request: DUUIMMRequest):
    model_name = request.model_name
    prompts = request.prompts
    responses_out, errors_out = [], []
    mode = request.mode

    try:
        if mode == MultiModelModes.TEXT:
            for prompt in prompts:
                responses_out.append(process_text_only(model_name, prompt))

        elif mode == MultiModelModes.IMAGE:
            if len(prompts) not in [1, len(request.images)]:
                errors_out.append("Number of prompts must be 1 or match number of images.")
            else:
                prompts = prompts * len(request.images) if len(prompts) == 1 else prompts
                for image, prompt in zip(request.images, prompts):
                    responses_out.append(process_image_only(model_name, image.src, prompt))

        else:
            errors_out.append(f"Mode {mode} is not supported in DUUI-Molmo yet.")

        return DUUIMMResponse(
            processed_text=responses_out,
            model_name=model_name,
            model_source=sources.get(model_name),
            model_lang=languages.get(model_name),
            model_version=versions.get(model_name),
            errors=errors_out,
            prompts=prompts
        )

    except Exception as ex:
        logger.exception(ex)
        return DUUIMMResponse(
            processed_text=[],
            model_name=model_name,
            model_source=sources.get(model_name),
            model_lang=languages.get(model_name),
            model_version=versions.get(model_name),
            errors=[str(ex)],
            prompts=prompts
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    uvicorn.run("duui-vision-language:app", host="0.0.0.0", port=9715, workers=1)
