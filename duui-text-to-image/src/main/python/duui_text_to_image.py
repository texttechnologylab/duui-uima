from diffusers import DiffusionPipeline
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from time import time
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
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
from starlette.responses import PlainTextResponse, JSONResponse
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

models_prompts_handler = {
    "OFA-Sys/small-stable-diffusion-v0": "handle_long_prompts",
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
typesystem_filename = 'TypeSystemTextToImage.xml'
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
    truncate_text: Optional[bool] = True



# Documentation response
class TextImagerDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str

    # Version of this annotator
    version: str

    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str]

    # Optional map of additional meta data
    meta: Optional[dict]

    # Optional map of supported parameters
    parameters: Optional[dict]



# Response sent by DUUI
# Note, this is transformed by the Lua script
class TextImagerResponse(BaseModel):
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
    config: Dict[str, Union[int, bool]]



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.text_to_image_annotator_name,
    description="Text To Image Component",
    version=settings.text_to_image_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Ali Abusaleh, TTLab",
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
    return TextImagerDocumentation(
        annotator_name=settings.text_to_image_annotator_name,
        version=settings.text_to_image_annotator_version,
        implementation_lang="Python",
        meta={"source": sources,
                "languages": languages,
                "versions": versions},
        parameters={
            "image_width": "Width of the generated images",
            "image_height": "Height of the generated images",
            "num_inference_steps": "Number of inference steps",
            "number_of_images": "Number of images to generate",
            "low_cpu_mem_usage": "Whether to use low CPU memory usage",
            "truncate_text": "Whether to truncate the text if it exceeds the model's maximum length",
        }
    )


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": ["string", "org.texttechnologylab.annotation.type.Image"],
        "outputs": ["string", "org.texttechnologylab.annotation.type.Image"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)



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



def fix_unicode_problems(text):
    # fix imgji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text



def check_and_tokenize_input(pipe, input_text, truncate):
    """
    Checks token sequence length and truncates it if the truncate flag is True.
    If truncate is False and sequence exceeds max length, returns an error.
    """
    max_length = pipe.tokenizer.model_max_length

    # Tokenize the input
    inputs = pipe.tokenizer(input_text, return_tensors="pt", padding=True)
    
    token_sequence_length = len(inputs['input_ids'][0])

    logger.debug(f"Token sequence length: {token_sequence_length}")

    if token_sequence_length > max_length and not truncate:
        return None, f"Input exceeds model max length ({token_sequence_length} > {max_length})"
    
    return inputs, None


def calculate_embeddings(pipe, prompt, negative_prompt="", truncate_text=False, device='cuda', pooling_strategy='mean'):
    """
    Calculate the embeddings of the prompt and negative prompt, and return both token embeddings and pooled embeddings.

    This function splits the prompt into chunks of the model's max token length, calculates embeddings for each chunk,
    applies pooling to the embeddings, and then returns both the raw token embeddings and the pooled embeddings.

    Args:
        pipe: The pipeline object containing the tokenizer and text encoder.
        prompt (str): The input text for which embeddings are calculated.
        negative_prompt (str, optional): The negative prompt (default is an empty string).
        truncate_text (bool, optional): Whether to truncate the text if it exceeds the model's maximum length (default is False).
        device (str, optional): The device to use for tensors ('cpu' or 'cuda', default is 'cuda').
        pooling_strategy (str, optional): The pooling strategy ('mean' or 'max') to apply to the embeddings (default is 'mean').

    Returns:
        tuple: A tuple containing:
            - prompt_embeds (torch.Tensor): The raw token embeddings for the prompt.
            - negative_prompt_embeds (torch.Tensor): The raw token embeddings for the negative prompt.
            - pooled_prompt_embeds (torch.Tensor): The pooled embeddings for the prompt.
            - pooled_negative_prompt_embeds (torch.Tensor): The pooled embeddings for the negative prompt.
    """

    # Tokenizer max length
    max_length = pipe.tokenizer.model_max_length

    # Tokenize input prompt and negative prompt
    input_ids = pipe.tokenizer(prompt, truncation=truncate_text, return_tensors="pt").input_ids.to(device)
    negative_ids = pipe.tokenizer(negative_prompt, truncation=truncate_text, padding="max_length",
                                  max_length=input_ids.shape[-1], return_tensors="pt").input_ids.to(device)

    # Split into chunks and calculate embeddings, then apply pooling
    prompt_embeds = _get_token_embeddings(pipe, input_ids, max_length)
    negative_prompt_embeds = _get_token_embeddings(pipe, negative_ids, max_length)

    # Apply pooling to get pooled embeddings
    pooled_prompt_embeds = _apply_pooling(prompt_embeds, pooling_strategy)
    pooled_negative_prompt_embeds = _apply_pooling(negative_prompt_embeds, pooling_strategy)

    # Ensure the embeddings are in the correct shape for the pipeline
    prompt_embeds = prompt_embeds.view(input_ids.shape[0], -1, prompt_embeds.shape[-1])  # Add batch dimension
    negative_prompt_embeds = negative_prompt_embeds.view(negative_ids.shape[0], -1, negative_prompt_embeds.shape[-1])

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds


def _get_token_embeddings(pipe, input_ids, max_length):
    """
    Helper function to split the input_ids into chunks, calculate the embeddings for each chunk.

    Args:
        pipe: The pipeline object containing the text encoder.
        input_ids (torch.Tensor): The tokenized input.
        max_length (int): The maximum length of each chunk.

    Returns:
        torch.Tensor: The raw token embeddings for the input.
    """
    # Initialize list to store embeddings for each chunk
    chunk_embeddings = []

    # Split input into chunks of max_length and calculate embeddings
    for i in range(0, input_ids.shape[-1], max_length):
        chunk = input_ids[:, i:i + max_length]
        chunk_embeddings.append(pipe.text_encoder(chunk)[0])  # The output of text_encoder is (embeddings, hidden_states)

    # Concatenate all the chunk embeddings
    return torch.cat(chunk_embeddings, dim=1)


def _apply_pooling(embeddings, pooling_strategy):
    """
    Apply pooling (mean or max) to the token embeddings to get a fixed-size representation.

    Args:
        embeddings (torch.Tensor): The token embeddings.
        pooling_strategy (str): The pooling strategy to use ('mean' or 'max').

    Returns:
        torch.Tensor: The pooled embeddings.
    """
    if pooling_strategy == 'mean':
        pooled_embeddings = torch.mean(embeddings, dim=1)  # Mean pooling across token length
    elif pooling_strategy == 'max':
        pooled_embeddings, _ = torch.max(embeddings, dim=1)  # Max pooling across token length
    else:
        raise ValueError(f"Invalid pooling strategy: {pooling_strategy}. Choose 'mean' or 'max'.")

    return pooled_embeddings



def process_selection(model_name, selection, model_config=None):
    """
    Process a selection of sentences and return the results.
    handle long prompts by splitting them into chunks of max_length
    """

    # Initialize output containers
    begin, end, results_out, factors, len_results, errors_list = [], [], [], [], [], []

    # Preprocess texts and fix unicode problems
    texts = [fix_unicode_problems(s.text) for s in selection.sentences]
    logger.debug("Preprocessed texts:")
    logger.debug(texts)

    try:
        with model_lock:
            # Load model and move it to the appropriate device (GPU/CPU)
            pipe = load_model(model_name, low_cpu_mem_usage=model_config["low_cpu_mem_usage"])
            logger.debug("Model loaded, starting inference")
            pipe.to(device)
            generator = torch.Generator("cuda").manual_seed(1024) if device == "cuda" else None

            for c, sentence in enumerate(selection.sentences):
                text = texts[c]
                inputs, error_message = check_and_tokenize_input(pipe, text, model_config["truncate_text"])
                # Initialize variables for this sentence
                begin.append(sentence.begin)
                end.append(sentence.end)
                factors.append([1.0])  # Dummy factor
                len_results.append(1)  # One image per text

                # Handle errors related to tokenization if the text is not truncated
                if error_message and not model_config["truncate_text"] and model_name in models_prompts_handler:
                    error_message += " The input will use embeddings for the prompts tokens."
                    errors_list.append(error_message)
                    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, pooled_negative_prompt_embeds = calculate_embeddings(pipe, text, truncate_text=False)
                    # Perform inference if there are no tokenization errors
                    results = pipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_pooled_prompt_embeds=pooled_negative_prompt_embeds,
                        num_inference_steps=model_config["num_inference_steps"],
                        num_images_per_prompt=model_config["number_of_images"],
                        image_width=model_config["image_width"],
                        image_height=model_config["image_height"],
                        generator=generator
                    )

                    results_out.append([image for image in results['images']])
                #     continue
                elif error_message and not model_config["truncate_text"] and model_name not in models_prompts_handler:
                    error_message += " Set truncate_text to True, or use a smaller prompt."
                    errors_list.append(error_message)
                    continue
                else:
                    # Perform inference if there are no tokenization errors
                    results = pipe(
                        prompt=text,
                        num_inference_steps=model_config["num_inference_steps"],
                        num_images_per_prompt=model_config["number_of_images"],
                        image_width=model_config["image_width"],
                        image_height=model_config["image_height"],
                        generator=generator
                    )
                    results_out.append([image for image in results['images']])





            logger.debug("Inference done")
            pipe.to("cpu")  # Free memory by moving the model back to CPU

    except Exception as ex:
        logger.exception(ex)
        errors_list.append(str(ex))

    # Return the processed output and model version
    return {
        "begin": begin,
        "end": end,
        "len_results": len_results,
        "results": results_out,
        "factors": factors,
        "errors": errors_list
    }, versions.get(model_name)


#

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: TextImagerRequest):
    # Fetch model-related information
    model_source = sources.get(request.model_name, "Unknown source")
    model_lang = languages.get(request.model_name, "Unknown language")
    model_version = versions.get(request.model_name, "Unknown version")

    # Configuration dictionary for model processing
    model_config = {
        "image_width": request.image_width,
        "image_height": request.image_height,
        "num_inference_steps": request.num_inference_steps,
        "number_of_images": request.number_of_images,
        "low_cpu_mem_usage": request.low_cpu_mem_usage,
        "truncate_text": request.truncate_text,
    }

    begin, end, len_results, results, factors, errors_list = [], [], [], [], [], []
    try:
        # Loop over selections in the request
        for selection in request.selections:
            # Process selection and extract results
            processed_sentences, _ = process_selection(request.model_name, selection, model_config=model_config)

            # Accumulate processed data
            begin.extend(processed_sentences["begin"])
            end.extend(processed_sentences["end"])
            len_results.extend(processed_sentences["len_results"])
            factors.extend(processed_sentences["factors"])
            errors_list.extend(processed_sentences["errors"])

            # Process the images in the current selection
            for image_list in processed_sentences["results"]:
                for image_obj in image_list:
                    # Convert image to base64 and add to results
                    buffered = BytesIO()
                    image_obj.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    result_image = Image(
                        src=img_str,
                        width=image_obj.size[0],
                        height=image_obj.size[1]
                    )
                    results.append(result_image)

                    # Free memory after processing image
                    image_obj.close()
                    del image_obj

        # Return the processed data in response
        return TextImagerResponse(
            begin_img=begin,
            end_img=end,
            results=results,
            len_results=len_results,
            factors=factors,
            model_name=request.model_name,
            model_version=model_version,
            model_source=model_source,
            model_lang=model_lang,
            errors=errors_list,
            config=model_config
        )

    except Exception as ex:
        logger.exception(ex)
        return TextImagerResponse(
            begin_img=[],
            end_img=[],
            results=[],
            len_results=[],
            factors=[],
            model_name=request.model_name,
            model_version=model_version,
            model_source=model_source,
            model_lang=model_lang,
            errors=[str(ex)],
            config=model_config
        )

    finally:
        # Free GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()




