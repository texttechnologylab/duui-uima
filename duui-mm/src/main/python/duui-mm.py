import base64
import logging
from functools import lru_cache
from http.client import responses
from threading import Lock
import io
import gc
import torch
import uvicorn
from PIL import Image
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from sympy import continued_fraction
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModelForVision2Seq
from models.duui_api_models import DUUIMMRequest, DUUIMMResponse, ImageType, Entity, Settings, DUUIMMDocumentation, MultiModelModes, LLMResult, LLMPrompt
from models.Phi_4_model import MicrosoftPhi4, Phi4ModelVLLM


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


# Global cache
_loaded_models = {}
_loaded_processors = {}


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=int(settings.mm_model_cache_size))

lua_communication_script, logger, type_system, device = None, None, None, None

def init():
    global lua_communication_script, logger, type_system, device


    logging.basicConfig(level=settings.mm_log_level)
    logger = logging.getLogger(__name__)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    logger.info(f'USING {device}')
    # Load the predefined typesystem that is needed for this annotator to work
    typesystem_filename = 'TypeSystemMM.xml'
    # logger.debug("Loading typesystem from \"%s\"", typesystem_filename)


    logger.debug("*"*20 + "Lua communication script" + "*"*20)
        # Load the Lua communication script
    lua_communication_script_filename = "duui-mm.lua"


    with open(lua_communication_script_filename, 'rb') as f:
        lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

    with open(typesystem_filename, 'rb') as f:
        type_system = load_typesystem(f)

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse, JSONResponse
model_lock = Lock()
sources = {
    "microsoft/Phi-4-multimodal-instruct": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct"
}

languages = {
    "microsoft/Phi-4-multimodal-instruct": "multi",
}

versions = {
    "microsoft/Phi-4-multimodal-instruct": "0af439b3adb8c23fda473c4f86001dbf9a226021",
}


init()

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.mm_annotator_name,
    description="DUUI component for Multimodal models",
    version=settings.mm_annotator_version,
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
            "versions": versions,
        },
        parameters={
            "prompt": "Prompt",
            "doc_lang": "Document language",
            "model_name": "Model name",
            "individual": "A flag for processing the images as one (set of frames) or indivisual. Note: it only works in a complex-mode",
            "mode": "a mode of operation"

        }
    )


@lru_cache_with_size
def load_model(model_name, device=None):
    """
    Load the model and optionally check the input sequence length if input_text is provided.
    Automatically truncates the input if it exceeds the model's max sequence length.
    """
    if model_name == "microsoft/Phi-4-multimodal-instruct":
        model = MicrosoftPhi4(logging_level=settings.mm_log_level)

    elif model_name == 'Phi4ModelVLLM':
        model = Phi4ModelVLLM()

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model



def process_text_only(model_name: str, prompt: LLMPrompt) -> LLMResult:
    model = load_model(model_name, device)
    response = model.process_text(prompt)
    return response  # Already an LLMResult from the model


def process_image_only(model_name: str, image_base64: str, prompt: LLMPrompt) -> LLMResult:
    model = load_model(model_name, device)
    result = model.process_image(image_base64, prompt)
    return result  # Already an LLMResult


def process_frames_only(model_name: str, frames: list[str], prompt: LLMPrompt) -> LLMResult:
    model = load_model(model_name, device)
    result = model.process_video_frames(prompt, frames)
    return result  # Already an LLMResult

def process_audio_only(model_name, audio_base64, prompt):
    model = load_model(model_name, device)
    response = model.process_audio(audio_base64, prompt)
    return response

def process_audio_video(model_name, audio_base64, frames_base64, prompt):
    model = load_model(model_name, device)
    response = model.process_video_and_audio(audio_base64, frames_base64, prompt)
    return response


def process_video_only(model_name, video_base64, prompt):
    model = load_model(model_name, device)

    response = model.process_video(video_base64, prompt)

    # TODO: postprocessing if needed
    return response



# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIMMRequest):
    # Fetch model-related information
    model_source = sources.get(request.model_name, "Unknown source")
    model_lang = languages.get(request.model_name, "Unknown language")
    model_version = versions.get(request.model_name, "Unknown version")

    prompts = request.prompts
    responses_out = []
    errors_out = []

    mode = request.mode
    individual = request.individual

    try:
        if mode == MultiModelModes.TEXT:
            for prompt in prompts:
                responses_out.append(process_text_only(request.model_name, prompt))

        elif mode == MultiModelModes.IMAGE or (mode == MultiModelModes.FRAMES and individual):
            if len(request.images) != len(prompts) and len(prompts) != 1:
                errors_out.append(
                    f"In {mode}, we need a prompt per image or 1 prompt for all images. "
                    f"Currently, we have {len(request.images)} images and {len(prompts)} prompts."
                )
            else:
                images = request.images if isinstance(request.images, list) else [request.images]
                if len(prompts) == 1:
                    prompts = [prompts[0]] * len(images)
                for image, prompt in zip(images, prompts):
                    responses_out.append(process_image_only(request.model_name, image.src, prompt))

        elif mode == MultiModelModes.FRAMES:
            if len(prompts) != 1:
                errors_out.append(
                    f"In {mode}, we need exactly 1 prompt for all frames. "
                    f"Currently, we have {len(prompts)} prompts."
                )
            else:
                result = process_frames_only(request.model_name, [img.src for img in request.images], prompts[0])
                responses_out.append(result)
        elif mode == MultiModelModes.AUDIO:
            if len(request.audio) != len(prompts) and len(prompts) != 1:
                errors_out.append(
                    f"In {mode}, we need a prompt per audio or 1 prompt for all audio inputs. "
                    f"Currently, {len(request.audio)} audio inputs and {len(prompts)} prompts."
                )
            else:
                audios = request.audio if isinstance(request.audio, list) else [request.audio]
                if len(prompts) == 1:
                    prompts = [prompts[0]] * len(audios)
                for audio, prompt in zip(audios, prompts):
                    responses_out.append(process_audio_only(request.model_name, audio, prompt))

        elif mode == MultiModelModes.FRAMES_AND_AUDIO:
            if len(prompts) != 1:
                errors_out.append(
                    f"In {mode}, we need exactly 1 prompt for the video+audio. "
                    f"Currently, {len(prompts)} prompts provided."
                )
            elif not request.audio or not request.images:
                errors_out.append("Both audio and image frames are required for AUDIO_VIDEO mode.")
            else:
                response = process_audio_video(
                    request.model_name,
                    request.audio[0] if isinstance(request.audio, list) else request.audio,
                    [img.src for img in request.images],
                    prompts[0]
                )
                responses_out.append(response)

        elif mode == MultiModelModes.VIDEO:
            if len(prompts) != 1 or len(request.videos) != 1:
                errors_out.append(f"In {mode}, exactly one prompt and one video must be provided. "
                                  f"Received {len(prompts)} prompts and {len(request.videos)} videos.")
            else:
                video_base64 = request.videos[0].src
                responses_out.append(process_video_only(request.model_name, video_base64, prompts[0]))
        else:
            raise Exception(f"Mode {mode}, is not supported.")


        # Return the final structured response
        return DUUIMMResponse(
            processed_text=responses_out,
            model_name=request.model_name,
            model_source=model_source,
            model_lang=model_lang,
            model_version=model_version,
            errors=errors_out,
            prompts=prompts
        )

    except Exception as ex:
        global logger
        logger.exception(ex)
        return DUUIMMResponse(
            processed_text=[],
            model_name=request.model_name,
            model_source=model_source,
            model_lang=model_lang,
            model_version=model_version,
            errors=[str(ex)],
            prompts=prompts
        )

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()



if __name__ == "__main__":
    uvicorn.run("duui_mm:app", host="0.0.0.0", port=9714, workers=1)

