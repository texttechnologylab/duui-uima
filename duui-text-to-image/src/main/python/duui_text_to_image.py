from diffusers import DiffusionPipeline
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from cassis import load_typesystem
import torch
from threading import Lock
from functools import lru_cache
from io import BytesIO
import base64
from huggingface_hub import login
import torch
import time
import gc

import warnings

# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse
model_lock = Lock()
sources = {
    "OFA-Sys/small-stable-diffusion-v0": "https://huggingface.co/OFA-Sys/small-stable-diffusion-v0",
    "stabilityai/stable-diffusion-xl-base-1.0": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
    "Shakker-Labs/Lumatales-FL": "https://huggingface.co/Shakker-Labs/Lumatales-FL",
    "RunDiffusion/Juggernaut-XL-v6": "https://huggingface.co/RunDiffusion/Juggernaut-XL-v6",
    "hassanelmghari/shou_xin": "https://huggingface.co/hassanelmghari/shou_xin",
}

languages = {
    "OFA-Sys/small-stable-diffusion-v0": "en",
    "stabilityai/stable-diffusion-xl-base-1.0": "en",
    "Shakker-Labs/Lumatales-FL": "en",
    "RunDiffusion/Juggernaut-XL-v6": "en",
    "hassanelmghari/shou_xin": "en",
}

versions = {
    "OFA-Sys/small-stable-diffusion-v0": "38e10e5e71e8fbf717a47a81e7543cd01c1a8140",
    "stabilityai/stable-diffusion-xl-base-1.0": "462165984030d82259a11f4367a4eed129e94a7b",
    "Shakker-Labs/Lumatales-FL": "8a07771494f995f4a39dd8afde023012195217a5",
    "RunDiffusion/Juggernaut-XL-v6": "3c3746c9e41e5543cd01e5f56c024d381ad11c2c",
    "hassanelmghari/shou_xin": "a1551631da706873a17c15e0ed0d266d8522655d",
}

lora_models = {
    "hassanelmghari/shou_xin": "hassanelmghari/shou_xin",
}

class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int


class UimaSentenceSelection(BaseModel):
    selection: str
    sentences: List[UimaSentence]

class Settings(BaseSettings):
    # Name of this annotator
    text_to_image_annotator_name: str
    # Version of this annotator
    # TODO add these to the settings
    text_to_image_annotator_version: str
    # Log level
    text_to_image_log_level: str
    # # # model_name
    # text_to_image_log_level: str
    # Name of this annotator
    text_to_image_model_version: str
    #cach_size
    text_to_image_model_cache_size: str

    # hugingface token
    text_to_image_hugging_face_token: str



# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=int(settings.text_to_image_model_cache_size))
logging.basicConfig(level=settings.text_to_image_log_level)
logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemTextToTimage.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_text_to_image.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)



class Image(BaseModel):
    """
    org.texttechnologylab.annotation.type.Image
    """
    src: str
    width: int
    height: int

# Request sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerRequest(BaseModel):
    # The texts language
    doc_len: int
    #
    lang: str
    #
    model_name: str
    #
    selections:  List[UimaSentenceSelection]

    # model configurations
    image_width: Optional[int] = 512
    image_height: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    number_of_images: Optional[int] = 1
    low_cpu_mem_usage: Optional[bool] = True




# Response sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerResponse(BaseModel):
    # Symspelloutput
    # List of Sentence with every token
    # Every token is a dictionary with following Infos:
    # Symspelloutput right if the token is correct, wrong if the token is incorrect, skipped if the token was skipped, unkownn if token can corrected with Symspell
    # If token is unkown it will be predicted with BERT Three output pos:
    # 1. Best Prediction with BERT MASKED
    # 2. Best Cos-sim with Sentence-Bert and with perdicted words of BERT MASK
    # 3. Option 1 and 2 together
    # images: List[Image]
    begin_img: List[int]
    end_img: List[int]
    results: List[Image]
    factors: List
    len_results: List[int]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str
    errors: List[str]



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.text_to_image_annotator_name,
    description="Factuality annotator",
    version=settings.text_to_image_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "a.abusaleh@em.uni-frankfurt.de",
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
    # TODO rimgve cassis dependency, as only needed for typesystem at the moment?
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



def check_and_tokenize_input(pipe, input_text):
    """
    Checks the token sequence length and ensures it is within the model's allowed max sequence length.
    If the sequence exceeds the max length, truncates it.
    Returns the tokenized input and an error message if applicable.
    """
    # Tokenize the input with truncation enabled
    inputs = pipe.tokenizer(input_text, return_tensors="pt", truncation=False, padding=True)

    # Capture the token sequence length
    token_sequence_length = len(inputs['input_ids'][0])

    # Get the model's maximum token length
    max_sequence_length = pipe.tokenizer.model_max_length

    # Initialize error message
    error_message = None

    # Check if the token sequence length exceeds the model's maximum length
    if token_sequence_length > max_sequence_length:
        error_message = f"Token indices sequence length is longer than the specified maximum sequence length for this model ({token_sequence_length} > {max_sequence_length})."
        # The truncation is already handled by the tokenizer, but you could also explicitly truncate here
        # Re-tokenize with truncation if it's not already done
        inputs = pipe.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

    return inputs, error_message

@lru_cache_with_size
def load_model(model_name, low_cpu_mem_usage, input_text=None):
    """
    Load the model and optionally check the input sequence length if input_text is provided.
    Automatically truncates the input if it exceeds the model's max sequence length.
    """
    if model_name in lora_models:
        # Login to Hugging Face (make sure the token is correct)
        login(token=settings['text_to_image_hugging_face_token'])

        # Load the lora model
        pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch.float16)
        pipe.load_lora_weights("hassanelmghari/shou_xin")
    else:
        # Load the specified model
        pipe = DiffusionPipeline.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch.float16)

    # Move the pipeline to the appropriate device
    pipe.to(device)

    return pipe


# @lru_cache_with_size
# def load_model(model_name, low_cpu_mem_usage):
#     if model_name in lora_models:
#         login(token=settings.text_to_image_hugging_face_token)
#         pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch.float16)
#         pipe.load_lora_weights("hassanelmghari/shou_xin")
#     else:
#         pipe = DiffusionPipeline.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage,  torch_dtype=torch.float16)
#
#     pipe.to(device)
#     return pipe


def fix_unicode_problems(text):
    # fix imgji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def process_selection(model_name, selection, doc_len, lang_document, model_config=None):
    begin = []
    end = []
    results_out = []
    factors = []
    len_results = []
    errors_list = []
    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    texts = [
        s.text
        for s in selection.sentences
    ]
    logger.debug("Preprocessed texts:")
    logger.debug(texts)

    with model_lock:
        pipe = load_model(model_name, low_cpu_mem_usage=model_config["low_cpu_mem_usage"])
        logger.debug("Model loaded, starting inference")
        # move model to gpu/cpu
        pipe.to(device)

        generator = torch.Generator("cuda").manual_seed(1024) if device == "cuda" else None
        for c, text in enumerate(texts):
            # Use catch_warnings to capture the warnings

            # Turn on the filter to always show warnings
            inputs, error_message = check_and_tokenize_input(pipe, text)

            if error_message:
                print(error_message)  # You can handle this as needed, like logging or raising an exception
            else:
                print("Input is within the allowed sequence length.")
                # Proceed as normal with the tokenized input
            results = pipe(text,
                           num_inference_steps=model_config["num_inference_steps"],
                           num_images_per_prompt=model_config["number_of_images"],
                           image_width=model_config["image_width"],
                           image_height=model_config["image_height"],
                           generator=generator)
            errors_list.append(error_message)
            for image in enumerate(results['images']):
                res_i = []
                factor_i = []
                sentence_i = selection.sentences[c]
                begin_i = sentence_i.begin
                end_i = sentence_i.end
                len_rel = 1  # Since we are generating one image per text
                begin.append(begin_i)
                end.append(end_i)
                res_i.append(image)
                factor_i.append(1.0)  # Dummy factor for image generation
                len_results.append(len_rel)
                results_out.append(res_i)
                factors.append(factor_i)
        logger.debug("Inference done")
        # move model to cpu to free memory
        pipe.to("cpu")

        # print("Memory cleaned")
        # for c, image in enumerate(results['images']):
        #     res_i = []
        #     factor_i = []
        #     sentence_i = selection.sentences[c]
        #     begin_i = sentence_i.begin
        #     end_i = sentence_i.end
        #     len_rel = 1  # Since we are generating one image per text
        #     begin.append(begin_i)
        #     end.append(end_i)
        #     res_i.append(image)
        #     factor_i.append(1.0)  # Dummy factor for image generation
        #     len_results.append(len_rel)
        #     results_out.append(res_i)
        #     factors.append(factor_i)
    output = {
        "begin": begin,
        "end": end,
        "len_results": len_results,
        "results": results_out,
        "factors": factors,
        "errors": errors_list
    }

    return output, versions[model_name]

#

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):

    model_source = sources[request.model_name]
    model_lang = languages[request.model_name]
    model_version = versions[request.model_name]
    # configurations
    model_config = {
    "image_width" : request.image_width ,
    "image_height" : request.image_height,
    "num_inference_steps" : request.num_inference_steps,
    "number_of_images" : request.number_of_images,
    "low_cpu_mem_usage": request.low_cpu_mem_usage,
    }

    begin = []
    end = []
    len_results = []
    results = []
    factors = []
    errors_list = []
    # Save modification start time for later
    try:
        model_source = sources[request.model_name]
        model_lang = languages[request.model_name]
        model_version = versions[request.model_name]
        lang_document = request.lang

        for selection in request.selections:
            processed_sentences, model_version_2 = process_selection(request.model_name, selection, request.doc_len, lang_document, model_config=model_config)
            begin = begin + processed_sentences["begin"]
            end = end + processed_sentences["end"]
            len_results = len_results + processed_sentences["len_results"]
            idx = 0
            for image in processed_sentences["results"]:
                idx = image[0][0]
                image = image[0][1]
                # image.save(f"original_{image[idx}.png")
                # idx += 1
                # Convert image to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                result_image = Image(
                    src=img_str,
                    width=image.size[0],
                    height=image.size[0]
                )
                # free memory
                image.close()
                del image
                results.append(result_image)
            # results = results + processed_sentences["results"]
            factors = factors + processed_sentences["factors"]
            errors_list = errors_list + processed_sentences["errors"]
    except Exception as ex:
        logger.exception(ex)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    return TextImagerResponse(begin_img=begin, end_img=end, results=results,
                              len_results=len_results, factors=factors,
                              model_name=request.model_name, model_version=model_version,
                              model_source=model_source, model_lang=model_lang, errors=errors_list)



