from typing import List, Optional
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging
import duui_logging
from duui_logging import log_info, log_warn, log_error, log_debug


# Settings
class Settings(BaseSettings):
    # Name of annotator
    annotator_name: str

    # Version of annotator
    annotator_version: str

    # Log level
    log_level: Optional[str] = None

    model_config = SettingsConfigDict(env_prefix='textimager_duui_vader_sentiment_')


# Capabilities
class TextImagerCapability(BaseModel):
    # List of supported languages by the annotator
    supported_languages: List[str]

    # Are results on same inputs reproducible without side effects?
    reproducible: bool


# Documentation response
class TextImagerDocumentation(BaseModel):
    # Name of this annotator
    annotator_name: str

    # Version of this annotator
    version: str

    # Annotator implementation language (Python, Java, ...)
    implementation_lang: Optional[str] = None

    # Optional map of additional meta data
    meta: Optional[dict] = None

    # Docker container id, if any
    docker_container_id: Optional[str] = None

    # Optional map of supported parameters
    parameters: Optional[dict] = None

    # Capabilities of this annotator
    capability: TextImagerCapability

    # Analysis engine XML, if available
    implementation_specific: Optional[str] = None