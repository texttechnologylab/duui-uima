
from pydantic import BaseModel
from enum import Enum


class ImageType(BaseModel):
    src: str
    height: int
    width: int
    begin: int
    end: int

class RedactType(str, Enum):
    blur = "blur"
    pixel = "pixel"
    black = "black"

class AnonType(str, Enum):
    single_align = "single_align"
    multiple_align = "multiple_align"
    swap = "swap"
    redact = "redact"


class DUUIRequest(BaseModel):
    anon_type: AnonType
    anon_degree: float
    images: List[ImageType]
    redact_type: RedactType
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
    output_images: List[ImageType]
    out_errors : List[str]

