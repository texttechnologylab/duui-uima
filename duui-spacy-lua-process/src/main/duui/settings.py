from typing import Annotated, Final, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from duui.const import (
    SPACY_LOOKUP,
    SpacyLanguage,
    SpacyModelName,
    SpacyModelSize,
    SpacyPipelineComponent,
)
from duui.errors import InvalidConfigurationError, NoModelError


class SpacySettings(BaseSettings):
    spacy_language: Annotated[SpacyLanguage, Field(examples=["de"])] = "de"
    spacy_model_size: Annotated[SpacyModelSize, Field(default="lg")]
    spacy_batch_size: Annotated[int, Field(default=32)] = 32

    spacy_model: Optional[
        Annotated[SpacyModelName, Field(examples=["de_core_news_lg"])]
    ] = None
    spacy_exclude: Optional[
        Annotated[
            list[SpacyPipelineComponent],
            Field(default_factory=list, examples=[[]]),
        ]
    ] = []
    spacy_disable: Optional[
        Annotated[
            list[SpacyPipelineComponent],
            Field(default_factory=lambda: ["senter"], examples=[["senter"]]),
        ]
    ] = []

    def resolve_model(self) -> SpacyModelName:
        if self.spacy_model:
            return self.spacy_model
        if self.spacy_language and self.spacy_model_size:
            model = SPACY_LOOKUP[self.spacy_model_size].get(self.spacy_language)
            if not model:
                raise InvalidConfigurationError(
                    f"Model not found for language {self.spacy_language} and mode {self.spacy_model_size}. "
                    "Please provide a valid model name or check the configuration."
                )
            return model
        raise NoModelError(
            "Invalid configuration: either 'model' or 'language' and 'mode' must be provided."
        )


class AppSettings(SpacySettings):
    component_name: str = "duui-spacy-v2"
    component_version: str = "0.1.0"
    max_loaded_models: int = 1
    request_batch_size: int = 1024


SETTINGS: Final[AppSettings] = AppSettings()
