import base64
import logging
import os
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import List, Optional
import io
import cv2
import gc
import numpy as np
import torch
import uvicorn
from PIL import Image
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from torchvision.transforms import functional as T
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModelForVision2Seq
# new
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').disabled = True

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


def plot_bbox(image, entities):
    """Draw high-quality bounding boxes with smart label placement using OpenCV."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))[:, :, ::-1]  # RGB to BGR

    image_h, image_w = image.shape[:2]
    image_copy = image.copy()
    used_label_boxes = []

    for entity in entities:
        if not entity.bounding_box:
            continue

        x1, y1, x2, y2 = map(int, entity.bounding_box[0])
        label = entity.name
        color = tuple(np.random.randint(0, 255, size=3).tolist())

        # Draw the bounding box
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness=2)

        # Label settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Get label size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_pad = 4
        box_w = text_w + 2 * label_pad
        box_h = text_h + 2 * label_pad

        # Default label position: above the bbox
        label_x1 = x1
        label_y1 = y1 - box_h if y1 - box_h > 0 else y1 + 2
        label_x2 = label_x1 + box_w
        label_y2 = label_y1 + box_h

        # Clamp horizontally
        if label_x2 > image_w:
            label_x1 = image_w - box_w
            label_x2 = image_w

        # Avoid overlapping with other labels
        max_attempts = 10
        attempts = 0
        while any(overlaps((label_x1, label_y1, label_x2, label_y2), box) for box in used_label_boxes) and attempts < max_attempts:
            label_y1 += box_h + 2
            label_y2 += box_h + 2
            if label_y2 > image_h:
                label_y1 = max(0, y1 - box_h)  # fallback to above box
                label_y2 = label_y1 + box_h
                break
            attempts += 1

        used_label_boxes.append((label_x1, label_y1, label_x2, label_y2))

        # Draw label background
        cv2.rectangle(image_copy, (label_x1, label_y1), (label_x2, label_y2), color, -1)

        # Draw label text
        text_x = label_x1 + label_pad
        text_y = label_y2 - label_pad
        cv2.putText(image_copy, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return Image.fromarray(image_copy[:, :, ::-1])  # Convert BGR to RGB


def overlaps(boxA, boxB, margin=4):
    """Check if two rectangles overlap, with a small margin."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 + margin < bx1 or ax1 - margin > bx2 or ay2 + margin < by1 or ay1 - margin > by2)




def custom_post_process_generation(task_prompt, image=None, text_input=None):
    # Free GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    model_id = 'microsoft/Florence-2-large'

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load model and convert to half precision on CUDA
    model = load_model(model_id, low_cpu_mem_usage=True)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch_dtype)

    if image is not None:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL.Image.Image, got: {type(image)}")
        image = image.convert("RGB")  # Ensure consistent format

    # Prepare the prompt
    prompt = task_prompt if text_input is None else task_prompt + text_input

    # Processor returns float32 by default — keep it that way
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda:0' if torch.cuda.is_available() else 'cpu', torch_dtype)

    # Generate
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=4096,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Post-process
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer




# Settings
# These are automatically loaded from env variables
from starlette.responses import PlainTextResponse, JSONResponse
model_lock = Lock()
sources = {
    "microsoft/kosmos-2-patch14-224": "https://huggingface.co/microsoft/kosmos-2-patch14-224",
    "microsoft/Phi-4-multimodal-instruct": "https://huggingface.co/microsoft/Phi-4-multimodal-instruct"
}

languages = {
    "microsoft/kosmos-2-patch14-224": "en",
    "microsoft/Phi-4-multimodal-instruct": "multi",
}

versions = {
    "microsoft/kosmos-2-patch14-224": "e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c",
    "microsoft/Phi-4-multimodal-instruct": "0af439b3adb8c23fda473c4f86001dbf9a226021",
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
typesystem_filename = '../resources/TypeSystemImageToText.xml'
# logger.debug("Loading typesystem from \"%s\"", typesystem_filename)

with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    # logger.debug("Base typesystem:")
    # logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_image_to_text.lua"
# logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)



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

    # individual or multiple image processing
    individual: bool = True





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
    # original prompt
    prompt: Optional[str] = None



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
    if model_name == "microsoft/kosmos-2-patch14-224":

        # Load the specified model
        model = AutoModelForVision2Seq.from_pretrained(model_name,
                                                       low_cpu_mem_usage=low_cpu_mem_usage,
                                                       torch_dtype=torch.float16)

    elif model_name == "microsoft/Phi-4-multimodal-instruct":
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     trust_remote_code=True,
                                                     _attn_implementation='flash_attention_2',
                                                     torch_dtype="auto")

    elif model_name == "microsoft/Florence-2-large":
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype="auto").eval()

    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    # Move the pipeline to the appropriate device
    model.to(device)

    return model



def fix_unicode_problems(text):
    # fix imgji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text

def convert_base64_to_image(base64_string):
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def find_label_positions(text, label):
    start = text.find(label)
    end = start + len(label) if start != -1 else -1
    return start, end

def process_complex(model_name, images, prompt):

    entities_all = []
    result_images_all = []

    # Load the model
    model = load_model(model_name, low_cpu_mem_usage=True)
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_name, device=device, trust_remote_code=True)


    # convert images from base64 to PIL
    images = [convert_base64_to_image(image.src) for image in images]

    placeholder = "".join(f"<|image_{i}|>" for i in range(len(images)))

    messages = [
        {
            "role": "user",
            "content": (
                    placeholder
                    +  prompt # "Please describe or summarize the content of these images." +
            )
        }
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(prompt, images=images, return_tensors='pt').to('cuda:0')

    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.5,
        "do_sample": True,
    }

    generation_config = GenerationConfig.from_pretrained(model_name)


    generate_ids = model.generate(
        **inputs,
        **generation_args,
        generation_config=generation_config,
    )

    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]

    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    processed_text_all = response

    #TODO: change this last image
    image = images[-1]
    entities, processed_text, generated_grounding = generate_custom_entities(image, response)

    for entity in entities:
        entity_name, bboxes = entity

        # Get the character positions
        start, end = find_label_positions(processed_text_all, entity_name)

        # Convert relative bboxes to absolute
        absolute_bboxes = [
            (
                int(x1 ),
                int(y1 ),
                int(x2 ),
                int(y2)
            )
            for x1, y1, x2, y2 in bboxes
        ]

        # Add entity with updated position info
        entities_all.append(Entity(name=entity_name, begin=start, end=end, bounding_box=absolute_bboxes))

    image_results = plot_bbox(image, generated_grounding)

    image_base64 = convert_image_to_base64(image_results)
    image_width, image_height = image.size

    result_images_all.append(ImageType(src=f"{image_base64}", width=image_width, height=image_height, begin=0, end=len(processed_text_all)))

    return processed_text, entities_all, result_images_all


def generate_custom_entities(image, generated_text, return_gounding=False):

    first_task_prompt = '<DETAILED_CAPTION>'

    detailed_caption = custom_post_process_generation(first_task_prompt,  image.copy(), None)

    print("detailed caption: ", detailed_caption)

    generated_text = generated_text + " In general: " + detailed_caption[first_task_prompt]

    print("generated text: ", generated_text)

    second_task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

    processed_entities_text = custom_post_process_generation(second_task_prompt, image.copy(), generated_text)
    entities = processed_entities_text[second_task_prompt]

    return entities, generated_text, processed_entities_text if return_gounding else None

def process_image(model_name, image, prompt, complex=False):
    print("processing image, with prompt, ", prompt)
    # convert image from base64 to PIL
    image = convert_base64_to_image(image)

    # Load the model
    model = load_model(model_name, low_cpu_mem_usage=True)
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_name, device=device, trust_remote_code=True)

    # phi-4 model
    if complex:
        placeholder  = f"<|image_0|>"
        messages = [
            {
                "role": "user",
                "content": (
                        placeholder
                        + prompt
                )
            }
        ]
        prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # get the inputs for the model
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs.to(device)

    if complex:
        generation_config = GenerationConfig.from_pretrained(model_name)

        generation_args = {
            "max_new_tokens": 512,
            "temperature": 0.5,
            "do_sample": True,
        }

        generated_ids = model.generate(
            **inputs,
            **generation_args,
            generation_config=generation_config,
        )
    else:
        # generate the IDs
        generated_ids = model.generate(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inputs["image_embeds_position_mask"],
            use_cache=True,
            max_new_tokens=1024,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    generated_text = generated_text.replace(prompt, "")

    if complex:

        # Free GPU memory if available
        model.to("cpu")

        entities, processed_text, _ = generate_custom_entities(image.copy(), generated_text)

        print("entities: ", entities)

        entities_out = []
        for bbox, label in zip(entities['bboxes'], entities['labels']):
            # Get the character positions
            start, end = find_label_positions(processed_text, label)


            x1, y1, x2, y2 = bbox
            # Convert relative bboxes to absolute
            absolute_bboxes = [
                (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
            ]

            # Add entity with updated position info
            entities_out.append(Entity(name=label, begin=start, end=end, bounding_box=absolute_bboxes))
            entities = entities_out

    else:
        # By default, the generated  text is cleanup and the entities are extracted.
        processed_text, entities = processor.post_process_generation(generated_text)

    print("generated text: ", processed_text)
    # return the processed text and the entities, which are the bounding boxes
    return processed_text, entities

def handle_image_results(image_path, entities):
    result_entities = []
    image_obj = convert_base64_to_image(image_path)
    image_width, image_height = image_obj.size
    # draw the entities on the image and save it into a buffer
    image_entities = draw_entity_boxes_on_image(image_obj, entities, show=False, save_path=None)
    image_base64 = convert_image_to_base64(Image.fromarray(image_entities[:, :, [2, 1, 0]]))

    for entity in entities:
        entity_name, (start, end), bboxes = entity
        absolute_bboxes = [
            (
                int(x1 * image_width),
                int(y1 * image_height),
                int(x2 * image_width),
                int(y2 * image_height)
            )
            for x1, y1, x2, y2 in bboxes
        ]
        result_entities.append(Entity(name=entity_name, begin=start, end=end, bounding_box=absolute_bboxes))

    return result_entities, image_base64
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
    complex = request.model_name == "microsoft/Phi-4-multimodal-instruct"

    print("processing with 'individual' = ", request.individual)
    try:
        if complex and request.individual == False:
            processed_text, result_entities, result_images = process_complex(request.model_name, request.images, prompt)
        else:
            for image in request.images:
                image_path = image.src
                image_width = image.width
                image_height = image.height
                # process the image
                processed_text, entities = process_image(request.model_name, image_path, prompt, complex)
                if complex:
                    result_entities = entities
                    image_base64 = convert_image_to_base64(plot_bbox(convert_base64_to_image(image.src), entities))

                else:
                    # handle the image results
                    result_entities, image_base64 = handle_image_results(image_path, entities)

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
            errors=errors_list,
            prompt=prompt
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
            errors=[str(ex)],
            prompt=prompt
        )

    finally:
        # Free GPU memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    uvicorn.run("duui_image_to_text:app", host="0.0.0.0", port=9714, workers=1)

