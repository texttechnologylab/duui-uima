import base64
import logging
import os
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import List, Optional

import cv2
import gc
import numpy as np
import torch
from PIL import Image
from cassis import load_typesystem
from diffusers import DiffusionPipeline
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from huggingface_hub import login
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from torchvision.transforms import functional as T
from transformers import AutoProcessor, AutoModelForVision2Seq


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, entities, show=False, save_path=None):
    """_summary_
    Args:
        image (_type_): image or image path
        collect_entity_location (_type_): _description_
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    if len(entities) == 0:
        return image

    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 1
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    for entity_name, (start, end), bboxes in entities:
        for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
            # draw bbox
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

            for prev_bbox in previous_bboxes:
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                    text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                    text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                    y1 += (text_height + text_offset_original + 2 * text_spaces)

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

            cv2.putText(
                new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
    if save_path:
        pil_image.save(save_path)
    if show:
        pil_image.show()

    return new_image



# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse, JSONResponse
model_lock = Lock()
sources = {
    "microsoft/kosmos-2-patch14-224": "https://huggingface.co/microsoft/kosmos-2-patch14-224",
}

languages = {
    "microsoft/kosmos-2-patch14-224": "en",
}

versions = {
    "microsoft/kosmos-2-patch14-224": "e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c",
}


class Settings(BaseSettings):
    # Name of this annotator
    image_to_text_annotator_name: str
    # Version of this annotator
    # TODO add these to the settings
    image_to_text_annotator_version: str
    # Log level
    image_to_text_log_level: str
    # # # model_name
    # Name of this annotator
    image_to_text_model_version: str
    #cach_size
    image_to_text_model_cache_size: str

    # hugingface token
    image_to_text_hugging_face_token: str


# Documentation response
class ImageToTextDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str

    # Version of this annotator
    version: str

    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str]

    # Optional map of additional meta data
    meta: Optional[dict]

    # Docker container id, if any
    docker_container_id: Optional[str]

    # Optional map of supported parameters
    parameters: Optional[dict]


class ImageType(BaseModel):
    """
    org.texttechnologylab.annotation.type.Image
    """
    src: str
    width: int
    height: int
    begin: int
    end: int

class Entity(BaseModel):
    """
    Named bounding box entity
    name: entity name
    begin: start position
    end: end position
    bounding_box: list of bounding box coordinates
    """
    name: str
    begin: int
    end: int
    bounding_box: List[tuple[float, float, float, float]]


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=int(settings.image_to_text_model_cache_size))
logging.basicConfig(level=settings.image_to_text_log_level)
logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
logger.info(f'USING {device}')
# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = './resources/TypeSystemImageToText.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)

with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_image_to_text.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)



# Request sent by DUUI
# Note, this is transformed by the Lua script
class ImageToTextRequest(BaseModel):

    # list of images
    images: List[ImageType]
    # prompt
    prompt: str

    # number of images
    number_of_images: int

    # doc info
    doc_lang: str

    # model name
    model_name: str





# Response sent by DUUI
# Note, this is transformed by the Lua script
class ImageToTextResponse(BaseModel):
    # list of processed text
    processed_text: str
    # list of entities (bounding boxes)
    entities: List[Entity]
    # list of images with entities drawn on them
    images: List[ImageType]
    # number of images
    number_of_images: int
    # model source
    model_source: str
    # model language
    model_lang: str
    # model version
    model_version: str
    # model name
    model_name: str
    # list of errors
    errors: Optional[List[str]]



app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.image_to_text_annotator_name,
    description="Image to Text Annotator",
    version=settings.image_to_text_model_version,
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
    return ImageToTextDocumentation(
        annotator_name=settings.image_to_text_annotator_name,
        version=settings.image_to_text_model_version,
        implementation_lang="Python",
        meta={
            "log_level": settings.image_to_text_log_level,
            "model_version": settings.image_to_text_model_version,
            "model_cache_size": settings.image_to_text_model_cache_size,
            "hugging_face_token": settings.image_to_text_hugging_face_token,
            "models": sources,
            "languages": languages,
            "versions": versions,
        },
        parameters={
            "images": "List of images",
            "prompt": "Prompt",
            "number_of_images": "Number of images",
            "doc_lang": "Document language",
            "model_name": "Model name"
        }
    )


@lru_cache_with_size
def load_model(model_name, low_cpu_mem_usage, input_text=None):
    """
    Load the model and optionally check the input sequence length if input_text is provided.
    Automatically truncates the input if it exceeds the model's max sequence length.
    """
    # Load the specified model
    model = AutoModelForVision2Seq.from_pretrained(model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch.float16)

    # Move the pipeline to the appropriate device
    model.to(device)

    return model



def fix_unicode_problems(text):
    # fix imgji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text

def convertBase64ToImage(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def convertImageToBase64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def process_image(model_name, image, prompt):

    # convert image from base64 to PIL
    image = convertBase64ToImage(image)

    # Load the model
    model = load_model(model_name, low_cpu_mem_usage=True)
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_name, device=device)

    # get the inputs for the model
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs.to(device)

    # generate the IDs
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=128,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # By default, the generated  text is cleanup and the entities are extracted.
    processed_text, entities = processor.post_process_generation(generated_text)

    # return the processed text and the entities, which are the bounding boxes
    return processed_text, entities


#

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: ImageToTextRequest):
    # Fetch model-related information
    model_source = sources.get(request.model_name, "Unknown source")
    model_lang = languages.get(request.model_name, "Unknown language")
    model_version = versions.get(request.model_name, "Unknown version")

    # model prompt
    prompt = request.prompt

    # Initialize the response
    processed_text = ""
    result_entities = []
    result_images = []
    errors_list = []
    print("number of images ", len(request.images))
    try:
        for image in request.images:
            image_path = image.src
            image_width = image.width
            image_height = image.height

            # process the image
            processed_text, entities = process_image(request.model_name, image_path, prompt)
            # draw the entities on the image and save it into a buffer
            image_entities = draw_entity_boxes_on_image(convertBase64ToImage(image_path), entities, show=False, save_path=None)
            image_base64 = convertImageToBase64(Image.fromarray(image_entities[:, :, [2, 1, 0]]))

            for entity in entities:
                entity_name, (start, end), bboxes = entity
                bboxes = [(x1, y1, x2, y2) for x1, y1, x2, y2 in bboxes]
                result_entities.append(Entity(name=entity_name, begin=start, end=end, bounding_box=bboxes))
            # append results to lists
            print(processed_text)
            result_images.append(ImageType(src=f"{image_base64}", width=image_width, height=image_height, begin=0, end=len(processed_text)))
        # Return the processed data in response
        return ImageToTextResponse(
            processed_text=processed_text,
            entities=result_entities,
            images=result_images,
            number_of_images=int(request.number_of_images),
            model_name=request.model_name,
            model_source=model_source,
            model_lang=model_lang,
            model_version=model_version,
            errors=errors_list
        )

    except Exception as ex:
        logger.exception(ex)
        return ImageToTextResponse(
            processed_text=[],
            entities=[],
            images=[],
            model_name=request.model_name,
            model_source=model_source,
            model_lang=model_lang,
            model_version=model_version,
            errors=[str(ex)]
        )

    finally:
        # Free GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()




