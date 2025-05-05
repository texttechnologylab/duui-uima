import logging
from typing import Final, Literal, Optional, get_args

import spacy
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.datastructures import State
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from spacy import Language

LOGGING_CONFIG: Final[dict] = uvicorn.config.LOGGING_CONFIG
LOGGING_CONFIG["loggers"][""] = {
    "handlers": ["default"],
    "level": "INFO",
    "propagate": False,
}
logging.config.dictConfig(LOGGING_CONFIG)


##### Settings #####


SpacyLanguage = Literal[
    "ca",
    "zh",
    "hr",
    "da",
    "nl",
    "en",
    "fi",
    "fr",
    "de",
    "el",
    "it",
    "ja",
    "ko",
    "lt",
    "mk",
    "nb",
    "pl",
    "pt",
    "ro",
    "ru",
    "sl",
    "es",
    "sv",
    "uk",
    "xx",
    "x-unspecified",
]

SpacyModel = Literal[
    "ca_core_news_sm",
    "zh_core_web_sm",
    "hr_core_news_sm",
    "da_core_news_sm",
    "nl_core_news_sm",
    "en_core_web_sm",
    "fi_core_news_sm",
    "fr_core_news_sm",
    "de_core_news_sm",
    "el_core_news_sm",
    "it_core_news_sm",
    "ja_core_news_sm",
    "ko_core_news_sm",
    "lt_core_news_sm",
    "mk_core_news_sm",
    "nb_core_news_sm",
    "pl_core_news_sm",
    "pt_core_news_sm",
    "ro_core_news_sm",
    "ru_core_news_sm",
    "sl_core_news_sm",
    "es_core_news_sm",
    "sv_core_news_sm",
    "uk_core_news_sm",
    "xx_sent_ud_sm",
]

SPACY_MODEL_LOOKUP: Final[dict[SpacyLanguage, SpacyModel]] = {
    "ca": "ca_core_news_sm",  # Catalan
    "zh": "zh_core_web_sm",  # Chinese
    "hr": "hr_core_news_sm",  # Croatian
    "da": "da_core_news_sm",  # Danish
    "nl": "nl_core_news_sm",  # Dutch
    "en": "en_core_web_sm",  # English
    "fi": "fi_core_news_sm",  # Finnish
    "fr": "fr_core_news_sm",  # French
    "de": "de_core_news_sm",  # German
    "el": "el_core_news_sm",  # Greek
    "it": "it_core_news_sm",  # Italian
    "ja": "ja_core_news_sm",  # Japanese
    "ko": "ko_core_news_sm",  # Korean
    "lt": "lt_core_news_sm",  # Lithuanian
    "mk": "mk_core_news_sm",  # Macedonian
    "nb": "nb_core_news_sm",  # Norwegian Bokmal
    "pl": "pl_core_news_sm",  # Polish
    "pt": "pt_core_news_sm",  # Portugese
    "ro": "ro_core_news_sm",  # Romanian
    "ru": "ru_core_news_sm",  # Russian
    "sl": "sl_core_news_sm",  # Slovenian
    "es": "es_core_news_sm",  # Spanish
    "sv": "sv_core_news_sm",  # Swedish
    "uk": "uk_core_news_sm",  # Ukrainian
    "xx": "xx_sent_ud_sm",  # Multi-Language / Unknown Language
    "x-unspecified": "xx_sent_ud_sm",  # Multi-Language / Unknown Language
}


class SpacySettings(BaseSettings):
    spacy_language: SpacyLanguage = "xx"


class AppSettings(SpacySettings):
    component_name: str = "duui-spacy-eos"
    component_version: str = "0.1.0"


SETTINGS: Final[SpacySettings] = AppSettings()


##### Initialization #####


def load_model(state: State, settings: SpacySettings):
    language: SpacyLanguage = settings.spacy_language
    if language not in SPACY_MODEL_LOOKUP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language '{language}'. Supported languages are: {', '.join(get_args(SpacyLanguage))}",
        )
    if not hasattr(state, "model") or state.model.lang != language:
        state.model = spacy.load(SPACY_MODEL_LOOKUP[language])
        logging.getLogger(__name__).info(
            f"Loaded Model: {app.state.model.lang} ({app.state.model.meta['name']})"
        )
    return state.model


app = FastAPI()
if not hasattr(app.state, "model"):
    load_model(app.state, SETTINGS)


##### DUUI V1 Communication Layer #####

with open("lua/communication_layer.lua", "r") as f:
    LUA_COMMUNICATION_LAYER: Final[str] = f.read()


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return LUA_COMMUNICATION_LAYER


##### DUUI V1 Communication Layer #####

with open("resources/type_system.xml", "r") as f:
    TYPE_SYSTEM_XML: Final[str] = f.read()


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=TYPE_SYSTEM_XML, media_type="application/xml")


##### Models #####


class DuuiRequest(BaseModel):
    text: str
    config: Optional[SpacySettings] = None


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

    @classmethod
    def from_nlp(cls, nlp: Language):
        return cls(
            name=SETTINGS.component_name,
            version=SETTINGS.component_version,
            spacy_version=spacy.__version__,
            model_lang=nlp.lang,
            model_name=nlp.meta["name"],
            model_pipes=nlp.pipe_names,
            model_spacy_git_version=nlp.meta["spacy_git_version"],
            model_spacy_version=nlp.meta["spacy_version"],
            model_version=nlp.meta["version"],
        )


class AnnotationType(BaseModel):
    begin: int
    end: int


class DuuiResponse(BaseModel):
    metadata: AnnotationMeta
    sentences: list[AnnotationType]


##### DUUI V1 Process Endpoint #####


@app.post("/v1/process", description="DUUI API v1 process endpoint")
async def v1_process(params: DuuiRequest, request: Request) -> DuuiResponse:
    config: SpacySettings = params.config or SETTINGS

    nlp: Language = load_model(request.app.state, config)

    # Determine the appropriate sentence segmentation component
    # based on the loaded spaCy model
    if "senter" in nlp.pipe_names:
        eos_pipe = ["senter"]
    elif "parser" in nlp.pipe_names:
        eos_pipe = ["senter"]
        nlp.enable_pipe("senter")
    elif "sentencizer" in nlp.pipe_names:
        eos_pipe = ["sentencizer"]
    else:
        raise HTTPException(
            status_code=500,
            detail=f"spaCy model {nlp.meta['name']} does not have a sentence segmentation component",
        )

    # Enable only the sentence segmentation pipeline component
    with nlp.select_pipes(enable=eos_pipe):
        # (potentially) increase maximum input length
        nlp.max_length = len(params.text) + 1

        return DuuiResponse(
            metadata=AnnotationMeta.from_nlp(nlp),
            sentences=[
                AnnotationType(
                    begin=sent.start_char,
                    end=sent.end_char,
                )
                for sent in nlp(params.text).sents
            ],
        )
