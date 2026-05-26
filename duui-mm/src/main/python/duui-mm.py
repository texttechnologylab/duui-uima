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
from models.duui_api_models import DUUIMMRequest, DUUIMMResponse, ImageType, Entity, Settings, DUUIMMDocumentation, MultiModelModes, LLMResult, LLMPrompt, AudioType, VideoTypes
from models.Phi_4_model import VllmMicrosoftPhi4, TransformersMicrosoftPhi4
from models.Qwen_V2_5 import *
from models.Qwen_3 import *

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
    typesystem_filename = './TypeSystemMM.xml'
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
    "vllm/microsoft/Phi-4-multimodal-instruct": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
    "microsoft/Phi-4-multimodal-instruct": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct",
    "vllm/Qwen/Qwen2.5-VL-7B-Instruct": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "Qwen/Qwen2.5-VL-3B-Instruct": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
    "Qwen/Qwen2.5-VL-32B-Instruct": "https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen/Qwen2.5-VL-32B-Instruct-AWQ": "https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    "Qwen/Qwen2.5-VL-72B-Instruct": "https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ": "https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
}


languages = {
    "vllm/microsoft/Phi-4-multimodal-instruct": "multi",
    "microsoft/Phi-4-multimodal-instruct": "multi",
    "vllm/Qwen/Qwen2.5-VL-7B-Instruct": "multi",
    "Qwen/Qwen2.5-VL-7B-Instruct": "multi",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ": "multi",
    "Qwen/Qwen2.5-VL-3B-Instruct": "multi",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ": "multi",
    "Qwen/Qwen2.5-VL-32B-Instruct": "multi",
    "Qwen/Qwen2.5-VL-32B-Instruct-AWQ": "multi",
    "Qwen/Qwen2.5-VL-72B-Instruct": "multi",
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ": "multi",
}

versions = {
    "vllm/microsoft/Phi-4-multimodal-instruct": "0af439b3adb8c23fda473c4f86001dbf9a226021",
    "microsoft/Phi-4-multimodal-instruct": "0af439b3adb8c23fda473c4f86001dbf9a226021",
    "vllm/Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "Qwen/Qwen2.5-VL-7B-Instruct": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ": "536a35794df8831aa814970ee8f89eff577e7718",
    "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
    "Qwen/Qwen2.5-VL-3B-Instruct-AWQ": "e7b623934290c5a4da0ee3c6e1e57bfb6b5abbf2",
    "Qwen/Qwen2.5-VL-32B-Instruct": "7cfb30d71a1f4f49a57592323337a4a4727301da",
    "Qwen/Qwen2.5-VL-32B-Instruct-AWQ": "66c370b74a18e7b1e871c97918f032ed3578dfef",
    "Qwen/Qwen2.5-VL-72B-Instruct": "cd3b627433ac68e782b69d5f829355b3f34fb7f2",
    "Qwen/Qwen2.5-VL-72B-Instruct-AWQ": "c8b87d4b81f34b6a147577a310d7e75f0698f6c2",
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
    if model_name == "vllm/microsoft/Phi-4-multimodal-instruct":
        model = VllmMicrosoftPhi4(logging_level=settings.mm_log_level)

    elif model_name == 'microsoft/Phi-4-multimodal-instruct':
        model = TransformersMicrosoftPhi4(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "vllm/Qwen/Qwen2.5-VL-7B-Instruct":
        model = VllmQwen2_5VL(logging_level=settings.mm_log_level)

    # Add conditions for the new Qwen/Qwen2.5-VL models
    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VL_7B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ":
        model = Qwen2_5_VL_7B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-3B-Instruct":
        model = Qwen2_5_VL_3B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-3B-Instruct-AWQ":
        model = Qwen2_5_VL_3B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-32B-Instruct":
        model = Qwen2_5_VL_32B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-32B-Instruct-AWQ":
        model = Qwen2_5_VL_32B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
        model = Qwen2_5_VL_72B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-72B-Instruct-AWQ":
        model = Qwen2_5_VL_72B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    # Add conditions for the new Qwen/Qwen2.5-VL models
    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
        model = Qwen2_5_VL_7B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct-AWQ":
        model = Qwen2_5_VL_7B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-3B-Instruct":
        model = Qwen2_5_VL_3B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-3B-Instruct-AWQ":
        model = Qwen2_5_VL_3B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-32B-Instruct":
        model = Qwen2_5_VL_32B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-32B-Instruct-AWQ":
        model = Qwen2_5_VL_32B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-72B-Instruct":
        model = Qwen2_5_VL_72B_Instruct(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen2.5-VL-72B-Instruct-AWQ":
        model = Qwen2_5_VL_72B_Instruct_AWQ(version=versions.get(model_name), logging_level=settings.mm_log_level)

    # Add conditions for the new Qwen3 models
    elif model_name == "Qwen/Qwen3-32B":
        model = Qwen3_32B(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen3-14B":
        model = Qwen3_14B(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen3-8B":
        model = Qwen3_8B(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen3-4B":
        model = Qwen3_4B(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen3-1.7B":
        model = Qwen3_1_7B(version=versions.get(model_name), logging_level=settings.mm_log_level)

    elif model_name == "Qwen/Qwen3-0.6B":
        model = Qwen3_0_6B(version=versions.get(model_name), logging_level=settings.mm_log_level)

    else:
        raise ValueError(f"Model {model_name} is not supported.")

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

def process_audio_only(model_name, audio:AudioType, prompt):
    audio_base64 = audio.src
    model = load_model(model_name, device)
    response = model.process_audio(audio_base64, prompt)
    return response

def process_audio_video(model_name, audio:AudioType, frames_base64, prompt):
    model = load_model(model_name, device)
    audio_base64 = audio.src
    response = model.process_video_and_audio(audio_base64, frames_base64, prompt)
    return response


def process_video_only(model_name, video: VideoTypes, prompt):
    model = load_model(model_name, device)

    video_base64 = video.src

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
            if prompts:
                if len(request.audios) != len(prompts) and len(prompts) != 1:
                    errors_out.append(
                        f"In {mode}, we need a prompt per audio or 1 prompt for all audio inputs. "
                        f"Currently, {len(request.audio)} audio inputs and {len(prompts)} prompts."
                    )
            if len(request.audios) != len(prompts) and len(prompts) != 1:
                errors_out.append(
                    f"In {mode}, we need a prompt per audio or 1 prompt for all audio inputs. "
                    f"Currently, {len(request.audios)} audio inputs and {len(prompts)} prompts."
                )
            else:
                audios = request.audios if isinstance(request.audios, list) else [request.audios]
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
                    request.audios[0] if isinstance(request.audios, list) else request.audios,
                    [img.src for img in request.images],
                    prompts[0]
                )
                responses_out.append(response)

        elif mode == MultiModelModes.VIDEO:
            if len(prompts) != 1 or len(request.videos) != 1:
                errors_out.append(f"In {mode}, exactly one prompt and one video must be provided. "
                                  f"Received {len(prompts)} prompts and {len(request.videos)} videos.")
            else:
                if len(prompts) == 1:
                    prompts = [prompts[0]] * len(request.videos)
                for video, prompt in zip(request.videos, prompts):
                    responses_out.append(process_video_only(request.model_name, video, prompt))
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

