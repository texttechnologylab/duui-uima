from enum import Enum
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional


class MultiModelModes(str, Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    AUDIO_ONLY = "audio_only"
    FRAMES_ONLY = "frames_only"
    FRAMES_AND_AUDIO = "frames_and_audio"



class Settings(BaseSettings):
    # Name of this annotator
    duui_mm_annotator_name: str
    # Version of this annotator
    # TODO add these to the settings
    duui_mm_version: str
    # Log level
    duui_mm_log_level: str
    # # # model_name
    # Name of this annotator
    duui_mm_model_version: str
    #cach_size
    duui_mm_model_cache_size: str



# Documentation response
class DUUIMMDocumentation(BaseModel):
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



# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIMMRequest(BaseModel):

    # list of images
    images: List[ImageType]
    # List of prompt
    prompts: List[str]

    # number of images
    number_of_images: int

    # doc info
    doc_lang: str

    # model name
    model_name: str

    # individual or multiple image processing
    individual: bool = False

    # mode for complex
    mode: MultiModelModes = MultiModelModes.TEXT_ONLY





# Response sent by DUUI
# Note, this is transformed by the Lua script
class DUUIMMResponse(BaseModel):
    # list of processed text
    processed_text: List[str]
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
