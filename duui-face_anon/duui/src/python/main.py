from __future__ import annotations

import base64
from io import BytesIO
import logging
from typing import Optional, Dict, List

from pydantic import BaseModel
import face_alignment
import torch
import gc


from fastapi.encoders import jsonable_encoder
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModel
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse, PlainTextResponse
from diffusers import AutoencoderKL, DDPMScheduler
from custom_referencenet.referencenet.referencenet_unet_2d_condition import (
    ReferenceNetModel,
)
from diffusers import UNet2DConditionModel
from custom_referencenet.referencenet.pipeline_referencenet import (
    StableDiffusionReferenceNetPipeline,
)
from utils.anonymize_faces_in_image import anonymize_faces_in_image
from utils.redact_faces import redact_faces_in_image



# --- duui communication classes
logger: Optional[logging.Logger] = None
typesystem: Optional[str] = None
device: Optional[torch.device] = None
pipe: Optional[StableDiffusionReferenceNetPipeline] = None
generator: Optional[torch.Generator] = None
fa: Optional[face_alignment.FaceAlignment] = None



class ImageType(BaseModel):
    src: str
    height: int
    width: int
    begin: int
    end: int

class DUUIRequest(BaseModel):
    anon_type: str
    anon_degree: float
    images: Dict[int, ImageType]
    redact_type: str
    blur: int
    pixel: int
    diffusion_model: str
    clip_model: str
    seed: int
    guidance: float
    inference_steps: int
    vis_input: bool
    height: Optional[int] = None
    width: Optional[int] = None
    hf_token: str

class DUUIResponse(BaseModel):
    output_images: Dict[int, ImageType]
    out_errors : List[str]



# ===== All the different run options =====
def single_aligned_face(source_image,
                        inference_steps,
                        guidance_scale,
                        anonymization_degree,
                        height,
                        width, vis_input, generator)-> Image:
    """

    :param source_image: image to be anonymized
    :param inference_steps: number of inference steps
    :param guidance_scale: the guidance scale
    :param anonymization_degree: degree of anonymization
    :param height: output image height
    :param width: input image height
    :param vis_input: weather to visualize input-output next to another


    :return: anonymized image
    """
    # generate an image that anonymizes faces

    with torch.no_grad():
        anon_image = pipe(
            source_image=source_image,
            conditioning_image=source_image,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=width,
            height=height,
        ).images[0]
        if vis_input:
            return combine_images([anon_image, source_image])

    return anon_image

def multiple_aligned_face(
        source_image,
        image_size,
        inference_steps,
        guidance_scale,
        anonymization_degree,
        generator
)->Image:
    """

    :param source_image: image to be anonymized
    :param image_size: image resize
    :param inference_steps: number of inference steps
    :param guidance_scale:
    :param anonymization_degree:
    :return:
    """


    # generate an image that anonymizes faces
    anon_image = anonymize_faces_in_image(
        image=source_image,
        face_alignment=fa,
        pipe=pipe,
        generator=generator,
        face_image_size=image_size,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        anonymization_degree=anonymization_degree,
    )

    return anon_image

def swap_faces(
        source_image,
        conditioning_image,
        inference_steps,
        guidance_scale,
        anonymization_degree,
        width,
        height,
        vis_input,
        generator
    ):
    """
    
    :param source_image: image to be anonymized
    :param conditioning_image: face to swap with
    :param inference_steps: number of infrence steps
    :param guidance_scale: guidance scale 
    :param anonymization_degree: degree of anonymization
    :param width: output image width (if not vis True)
    :param height: output image height (if not vis True)
    :param vis_input: weather to visualize input-output next to another
    :return: 
    """""
    # generate an image that swaps faces
    assert pipe is not None
    with torch.no_grad():
        swap_image = pipe(
            source_image=source_image,
            conditioning_image=conditioning_image,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=width,
            height=height,
        ).images[0]
        if vis_input:
            return combine_images([swap_image, source_image])
    return swap_image


def redact_faces(
        source_image,
        image_size,
        redaction_method,
        blur_strength,
        pixel_size,
        vis_input,
    )->Image:
    """

    :param source_image: image to be redacted
    :param image_size: image size for resizing
    :param redaction_method: which method to choose
    :param blur_strength: blur strength for blurring
    :param pixel_size: pixel size for pixelation
    :param vis_input: weather to visualize input-output next to another

    :return:
    """
    redact_image = redact_faces_in_image(
        source_image=source_image,
        face_image_size=image_size,
        redaction_method=redaction_method,
        blur_strength=blur_strength,
        pixel_size=pixel_size,
    )
    if vis_input:
        return combine_images([redact_image, source_image])
    return redact_image




# ===== Helper

def combine_images(images):
    """
    Combines images for comparison of input-output.

    :param images:
    :return: One larger image with both original images placed next to another.
    """
    # Get the total width and maximum height of all images

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    # Create a new image with the combined width and maximum height
    new_image = Image.new("RGB", (total_width, max_height))

    # Paste each image onto the new image horizontally
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image


def pil_to_b64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def b64_to_pil(b64_str):
    img_bytes = base64.b64decode(b64_str)
    return Image.open(BytesIO(img_bytes)).convert("RGB")

def load_typesystem()-> str:
    #Load the predefined typesystem that is needed for this annotator to work
    typesystem_filename = 'resources/typesystem_face_anon.xml'
    with open(typesystem_filename, 'r') as f:
        type_system = f.read()

        logging.basicConfig(level=logging.INFO)
        logger.info("Loaded type system from %s", typesystem_filename)
        return type_system



def init():
    global logger, typesystem

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    typesystem = load_typesystem()


def load_pipeline(clip_model, diffusion_model, seed, token):
    global pipe, generator, fa
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
    # SFD (likely best results, but slower)
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, face_detector="sfd"
    )
    face_model_id = "hkung/face-anon-simple"
    unet = UNet2DConditionModel.from_pretrained(
        face_model_id, subfolder="unet", use_safetensors=True, token=token, torch_dtype=torch.float16
    )
    referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="referencenet", use_safetensors=True, token=token, torch_dtype=torch.float16
    )
    conditioning_referencenet = ReferenceNetModel.from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", use_safetensors=True, token=token, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        diffusion_model, subfolder="vae", use_safetensors=True,token=token, torch_dtype=torch.float16
    )
    scheduler = DDPMScheduler.from_pretrained(
        diffusion_model, subfolder="scheduler", use_safetensors=True, token=token)

    feature_extractor = CLIPImageProcessor.from_pretrained(
        clip_model, use_safetensors=True, token=token, torch_dtype=torch.float16
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        clip_model, use_safetensors=True, token=token, torch_dtype=torch.float16
    )


    pipe = StableDiffusionReferenceNetPipeline(
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)


# === the container
init()

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    title="duui-face_anon",
    description="Implementation of [WACV 2025] 'Face Anonymization Made Simple' for DUUI.",
    version="0.1",
        contact={
            "name": "Coco Sittardt, TTLab Team",
            "url": "https://texttechnologylab.org",
            "email": "sittardt@em.uni-frankfurt.de",
        },
        license_info={
            "name": "AGPL",
            "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
        },
)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error on {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )
@app.get("/v1/details/input_output")
def get_input_output()-> JSONResponse:
    json_item = {
       "inputs" : ["org.texttechnologylab.annotation.type.Image"],
        "outputs" : ["org.texttechnologylab.annotation.type.Image"]
    }
    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem,
        media_type="application/xml"
    )


# Load the Lua communication script
communication = "resources/communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")

# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication

@app.post("/v1/process")
def post_process(request:DUUIRequest)-> DUUIResponse:
    """


    """
    print(request)
    # the base selection between which anonymization is run
    anon_type = request.anon_type
    # the amount of anonymization
    anon_degree = request.anon_degree
    # input images
    images = request.images
    # set if the anon_type is redaction, then can choose again between blur, black or pixel
    redact_type = request.redact_type
    blur = request.blur
    pixel = request.pixel
    diffusion_model = request.diffusion_model
    clip_model = request.clip_model
    seed = request.seed
    guidance = request.guidance
    inference_steps = request.inference_steps
    vis_input = request.vis_input

  


    hf_token = request.hf_token


    output_images = {}
    errors_out =[]
    try:
        if len(images) == 0:
            raise ValueError("No Images provided")
        if hf_token == "None":
            raise ValueError("Please provide a hugging face token, to access the models.")

        load_pipeline(clip_model, diffusion_model, seed, hf_token)
        # selection between the different anon types:
        # options: single_align, multiple_align, swap, redact
        for img_id, img_data in images.items():
              # these can be "None" and will then be set later in the loop, UNLESS predefined height / width is passed
            height = request.height
            width = request.width

            source_image = b64_to_pil(img_data.src)
            if height is None:
                height = source_image.height
            if width is None:
                width = source_image.width

            MAX_DIM = 768
            if height > MAX_DIM or width > MAX_DIM:
                scale = MAX_DIM / max(height, width)
                height = (int(height * scale)//8)*8
                width = (int(width * scale)//8)*8
                source_image = source_image.resize((width, height), Image.LANCZOS)

            # only one image
            if anon_type == "single_align":
                generator = torch.Generator(device="cuda").manual_seed(seed)


                output = single_aligned_face(source_image, inference_steps=inference_steps,
                                             guidance_scale=guidance, anonymization_degree=anon_degree, height=height,
                                             width=width, vis_input=vis_input, generator=generator)
                output_images[img_id] = ImageType(
                    src=pil_to_b64(output),
                    height=height,
                    width=width,
                    begin = img_data.begin,
                    end = img_data.end,
                )

            elif anon_type == "multiple_align":
                generator = torch.Generator(device="cuda").manual_seed(seed)
                if height != width:

                    errors_out.append("width != height")

                output = multiple_aligned_face(
                    source_image=source_image,
                    image_size=height,
                    inference_steps=inference_steps,
                    guidance_scale=guidance,
                    anonymization_degree=anon_degree,
                    generator=generator
                )

                output_images[img_id] = ImageType(
                    src=pil_to_b64(output),
                    height=height,
                    width=width,
                    begin=img_data.begin,
                    end=img_data.end,
                )


            elif anon_type == "swap":
                generator = torch.Generator(device="cuda").manual_seed(seed)

                if len(images) != 2:
                    errors_out.append("To swap two faces an input of exactly two images is required.")
                    raise ValueError(
                        f"You have passed a total number of {len(images)} images. To swap you need to pass exactly 2.")
                ids = list(images.values()) # work around
                output = swap_faces(
                    source_image=b64_to_pil(ids[0].src),
                    conditioning_image=b64_to_pil(ids[1].src),
                    inference_steps=inference_steps,
                    guidance_scale=guidance,
                    anonymization_degree=anon_degree,
                    width=width,
                    height=height,
                    vis_input=vis_input,
                    generator=generator
                )
                # uses id 1, ignores id 2 just to be able to insert it better into the CAS
                output_images[1] = ImageType(
                    src=pil_to_b64(output),
                    height=height,
                    width=width,
                    begin=img_data.begin,
                    end=img_data.end
                )
                # can only run once so the iter across all images stops
                break
            elif anon_type == "redact":
                if redact_type == "None":
                    errors_out.append("Redaction Type has not been set - using default (blur)")
                    redact_type="blur"
                if redact_type == "blur" and blur%2==0:
                    errors_out.append(f"The passed blur parameter ({blur}) was even. Setting to default 51.")
                    blur = 51

                output = redact_faces(
                    source_image=source_image,
                    image_size=height,
                    redaction_method=redact_type,
                    blur_strength=blur,
                    pixel_size=pixel,
                    vis_input=vis_input
                )
                output_images[img_id] = ImageType(
                    src=pil_to_b64(output),
                    height=height,
                    width=width,
                    begin=img_data.begin,
                    end=img_data.end,
                )
            else:
                raise ValueError(f"Unknown anon_type: {anon_type}")


        return DUUIResponse(
            output_images=output_images,
            out_errors = errors_out
        )
    except Exception as ex:
        global logger
        logger.exception(ex)
        return DUUIResponse(
            output_images={},
            out_errors=[str(ex)],
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
