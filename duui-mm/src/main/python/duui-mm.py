import base64
import logging
from functools import lru_cache
from threading import Lock
import io
import cv2
import gc
import torch
import uvicorn
from PIL import Image
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoModelForVision2Seq
from models.duui_api_models import DUUIMMRequest, DUUIMMResponse, ImageType, Entity, Settings, DUUIMMDocumentation, MultiModelModes
from models.utils import convert_base64_to_image, convert_image_to_base64, draw_entity_boxes_on_image, find_label_positions, plot_bbox
from models.Phi_4_model import MicrosoftPhi4


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


# Global cache
_loaded_models = {}
_loaded_processors = {}


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=int(settings.image_to_text_model_cache_size))

lua_communication_script, logger, type_system, device = None, None, None, None, None, None

def init():
    global lua_communication_script, logger, type_system, device


    logging.basicConfig(level=settings.image_to_text_log_level)
    logger = logging.getLogger(__name__)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    logger.info(f'USING {device}')
    # Load the predefined typesystem that is needed for this annotator to work
    typesystem_filename = '../resources/TypeSystemMM.xml'
    # logger.debug("Loading typesystem from \"%s\"", typesystem_filename)


    print("*"*20, "Lua communication script", "*"*20)
        # Load the Lua communication script
    lua_communication_script_filename = "duui_mm.lua"


    with open(lua_communication_script_filename, 'rb') as f:
        lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

    with open(typesystem_filename, 'rb') as f:
        type_system = load_typesystem(f)







def custom_post_process_generation(task_prompt, image=None, text_input=None):
    # # Free GPU memory if available
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    #     gc.collect()

    model_id = 'microsoft/Florence-2-large'

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    florence_model, processor = get_model_and_processor(model_id)

    if image is not None:
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL.Image.Image, got: {type(image)}")
        image = image.convert("RGB")  # Ensure consistent format

    # Prepare the prompt
    prompt = task_prompt if text_input is None else task_prompt + text_input

    # Processor returns float32 by default — keep it that way
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda:0' if torch.cuda.is_available() else 'cpu', torch_dtype)

    # Generate
    generated_ids = florence_model.generate(
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


init()

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.duui_mm_annotator_name,
    description="Image to Text Annotator",
    version=settings.duui_mm_version,
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
            "model_name": "Model name",
            "individual": "A flag for processing the images as one (set of frames) or indivisual. Note: it only works in a complex-mode",
            "mode": "a mode of operation, simple --> Kosmos-2 model only, complex-only-answer --> phi-4 MM model only, complex-with-OD --> phi-4 mm followed by Florence-2 model"

        }
    )


@lru_cache_with_size
def load_model(model_name, device):
    """
    Load the model and optionally check the input sequence length if input_text is provided.
    Automatically truncates the input if it exceeds the model's max sequence length.
    """
    if model_name == "microsoft/Phi-4-multimodal-instruct":
        model = MicrosoftPhi4(device=device,
                              logging_level=settings.duui_mm_log_level)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model


@lru_cache_with_size
def get_model_and_processor(model_name):
    if model_name not in _loaded_models:
        print(f"Loading model: {model_name}")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _loaded_models[model_name] = load_model(model_name, low_cpu_mem_usage=True)
        _loaded_processors[model_name] = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch_dtype)
    return _loaded_models[model_name], _loaded_processors[model_name]






def process_complex(model_name, images, prompt, mode):

    entities_all = []
    result_images_all = []

    # Load the model
    model, processor = get_model_and_processor(model_name)


    # convert images from base64 to PIL
    images = [convert_base64_to_image(image.src) for image in images]

    placeholder = "".join(f"<|image_{i}|>" for i in range(len(images)))

    messages = [
        {
            "role": "user",
            "content": f"{placeholder},Please describe or summarize the content of these images as they are a series of frames representa video. AND based on the the description, {prompt}"
        }
    ]


    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("processed prompt is ", prompt)

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

    generation_args = {
        "max_new_tokens": 4096,
        "return_full_text": False,
        "temperature": 0.00001,
        "top_p": 1.0,
        "do_sample": True,
    }

    response = processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
        **generation_args
    )[0]
    print("the answer is: ", response)
    processed_text_all = response

    detail = mode == ComplexOperatingMode.COMPLEX_WITH_OD

    for image in images:
        # #TODO: change this last image
        # image = images[-1]
        entities, processed_text, generated_grounding = generate_custom_entities(image, response, detail=detail)
        sub_entities= []
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
            sub_entities.append(Entity(name=label, begin=start, end=end, bounding_box=absolute_bboxes))


        image_results = plot_bbox(image, sub_entities)

        entities_all.extend(sub_entities)

        image_base64 = convert_image_to_base64(image_results)
        image_width, image_height = image.size

        result_images_all.append(ImageType(src=f"{image_base64}", width=image_width, height=image_height, begin=0, end=len(processed_text_all)))

    return processed_text_all, entities_all, result_images_all


def generate_custom_entities(image, generated_text, return_gounding=False, detail =True):
    print("processing entities with detail? ", detail)

    if detail:
        first_task_prompt = '<DETAILED_CAPTION>'

        detailed_caption = custom_post_process_generation(first_task_prompt,  image.copy(), None)

        print("detailed caption: ", detailed_caption)

        generated_text = generated_text + " In general: " + detailed_caption[first_task_prompt]


    print("generated text: ", generated_text)

    second_task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

    processed_entities_text = custom_post_process_generation(second_task_prompt, image.copy(), generated_text)
    entities = processed_entities_text[second_task_prompt]

    return entities, generated_text, processed_entities_text if return_gounding else None

def process_image(model_name, image, prompt, mode):
    print("processing image, with prompt, ", prompt)

    # Load the model
    model = load_model(model_name, low_cpu_mem_usage=True)


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

        # # Free GPU memory if available
        # model.to("cpu")

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

def process_text_only(model_name, prompt):
    # Load the model
    model = load_model(model_name, low_cpu_mem_usage=True)

    response = model.process_text(prompt)

    #TODO: postprocessing

    # return the processed text
    return response

def process_image_only(model_name, image_base64, prompt):
    # Load the model
    model = load_model(model_name, low_cpu_mem_usage=True)

    generated_text = model.process_image(image_base64, prompt)

    # return the processed text
    return generated_text

# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIMMRequest):
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

    mode = request.mode
    individual = request.individual
    try:
        if mode == MultiModelModes.TEXT_ONLY:
            response = process_text_only(request.model_name, prompt)
        elif mode == MultiModelModes.IMAGE_ONLY:
            for image in request.images if type(request.images) == list else [request.images]:
                responses = [process_image_only(request.model_name, image.src, prompt)]


        for image in request.images:
            image_path = image.src
            image_width = image.width
            image_height = image.height
            # process the image
            processed_text, entities = process_image(request.model_name, image_path, prompt, mode)
            if complex:
                result_entities = entities
                image_base64 = convert_image_to_base64(plot_bbox(convert_base64_to_image(image.src), entities))

            else:
                # handle the image results
                result_entities, image_base64 = handle_image_results(image_path, entities)

            result_images.append(ImageType(src=f"{image_base64}", width=image_width, height=image_height, begin=0, end=len(processed_text)))
        # Return the processed data in response
        return DUUIMMResponse(
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
        return DUUIMMResponse(
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
    uvicorn.run("duui_mm:app", host="0.0.0.0", port=9714, workers=1)

