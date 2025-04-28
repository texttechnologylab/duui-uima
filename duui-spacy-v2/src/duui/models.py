from typing import Annotated, Optional

from pydantic import BaseModel, Field

from duui.settings import SpacySettings


class AnnotationMeta(BaseModel):
    name: str
    version: str
    spacy_version: str
    model_name: str
    model_pipes: list[str]
    model_version: str
    model_lang: str
    model_spacy_version: str
    model_spacy_git_version: str


class AnnotationType(BaseModel):
    begin: int
    end: int


class TokenType(AnnotationType):
    lemma: Optional[str] = None
    pos_value: Optional[str] = None
    pos_coarse: Optional[str] = None
    morph_value: Optional[str] = None
    morph_features: Optional[dict[str, str]] = None


class DependencyType(AnnotationType):
    dependency_type: str
    flavor: str = "basic"
    governor_index: int
    dependent_index: int


class EntityType(AnnotationType):
    value: str
    identifier: Optional[str] = None


class DuuiResponse(BaseModel):
    metadata: AnnotationMeta
    tokens: Optional[list[TokenType]] = None
    dependencies: Optional[list[DependencyType]] = None
    entities: Optional[list[EntityType]] = None


class ComponentCapability(BaseModel):
    # List of supported languages by the annotator
    # - ISO 639-1 (two letter codes) as default in meta data
    # - ISO 639-3 (three letters) optionally in extra meta to allow a finer mapping
    supported_languages: list[str]
    # Are results on same inputs reproducible without side effects?
    reproducible: bool


class ComponentDocumentation(BaseModel):
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
    capability: ComponentCapability
    # Analysis engine XML, if available
    implementation_specific: Optional[str] = None


class DuuiSentence(BaseModel):
    text: str
    offset: int


class DuuiRequest(BaseModel):
    sentences: Annotated[
        list[DuuiSentence],
        Field(
            examples=[
                [
                    {
                        "text": "Die Goethe Universität ist auf vier große Universitätsgelände über das Frankfurter Stadtgebiet verteilt.",
                        "offset": 0,
                    },
                    {
                        "offset": 1000,
                        "text": "Barack Obama war von 2009 bis 2017 der 44. Präsident der Vereinigten Staaten.",
                    },
                ]
            ]
        ),
    ]
    config: Optional[SpacySettings] = None


class EosRequest(BaseModel):
    text: str
    config: Optional[SpacySettings] = None


class EosResponse(BaseModel):
    metadata: AnnotationMeta
    sentences: list[AnnotationType]
