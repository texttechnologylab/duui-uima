
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ImageType(BaseModel):
    src: str
    height: int
    width: int
    begin: int
    end: int


class VideoType(BaseModel):
    src: str
    length: float = -1
    fps: float = -1
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


class SamplingMode(str, Enum):
    uniform = "uniform"
    adaptive = "adaptive"


class DUUIRequest(BaseModel):
    anon_type: AnonType
    anon_degree: float
    images: List[ImageType] = Field(default_factory=list)
    videos: List[VideoType] = Field(default_factory=list)
    sampling_mode: SamplingMode = SamplingMode.uniform
    frame_interval: int = Field(default=1, ge=1)
    segment_duration: float = Field(default=10.0, gt=0)
    representative_frames: int = Field(default=5, ge=1)
    redact_type: RedactType
    face_type: str = "full_face"
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
    output_images: List[ImageType] = Field(default_factory=list)
    output_videos: List[VideoType] = Field(default_factory=list)
    out_errors: List[str]
