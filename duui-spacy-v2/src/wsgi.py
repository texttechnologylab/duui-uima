import logging
from platform import python_version
from sys import version as sys_version
from typing import Final, get_args

import spacy
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.logger import logger
from fastapi.responses import PlainTextResponse
from spacy import Language

from duui.const import (
    LUA_COMMUNICATION_LAYER,
    SpacyLanguage,
    SpacyModelName,
    SpacyPipelineComponent,
)
from duui.errors import NoModelError
from duui.models import (
    AnnotationMeta,
    AnnotationType,
    ComponentCapability,
    ComponentDocumentation,
    DependencyType,
    DuuiRequest,
    DuuiResponse,
    EntityType,
    EosRequest,
    EosResponse,
    TokenType,
)
from duui.settings import SETTINGS, SpacySettings
from duui.utils import (
    get_spacy_model,
)

LOGGING_CONFIG: Final[dict] = uvicorn.config.LOGGING_CONFIG
LOGGING_CONFIG["loggers"][""] = {
    "handlers": ["default"],
    "level": "INFO",
    "propagate": False,
}
logging.config.dictConfig(LOGGING_CONFIG)

app = FastAPI()
if not hasattr(app.state, "models"):
    app.state.models = {}
if not hasattr(app.state, "lru"):
    app.state.lru = []
try:
    get_spacy_model(app.state, SETTINGS)
except NoModelError:
    pass


#####


@app.get(
    "/v1/communication_layer",
    response_class=PlainTextResponse,
    description="DUUI API v1: Get the Lua communication layer",
)
def get_communication_layer() -> str:
    with open("communication_layer.lua", "r") as f:
        return f.read()
    return LUA_COMMUNICATION_LAYER


#####


@app.get("/v1/documentation")
def get_documentation(request: Request) -> ComponentDocumentation:
    return ComponentDocumentation(
        annotator_name=SETTINGS.component_name,
        version=SETTINGS.component_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "spacy_version": spacy.__version__,
        },
        parameters={
            "spacy_model": list(get_args(SpacyModelName)),
            "spacy_language": list(get_args(SpacyLanguage)),
            "spacy_mode": ["accuracy", "efficiency"],
            "spacy_exclude": list(get_args(SpacyPipelineComponent)),
            "spacy_disable": list(get_args(SpacyPipelineComponent)),
            "spacy_batch_size": "int",
        },
        capability=ComponentCapability(
            supported_languages=sorted(list(get_args(SpacyLanguage))), reproducible=True
        ),
        implementation_specific=None,
    )


#####


@app.post("/v1/process", description="DUUI API v1 process endpoint")
async def v1_process(
    params: DuuiRequest,
    request: Request,
) -> DuuiResponse:
    config: SpacySettings = params.config or SETTINGS

    nlp: Language = get_spacy_model(request.app.state, config)

    to_disable = config.spacy_disable or SETTINGS.spacy_disable or []
    to_disable = set(to_disable).intersection(nlp.pipe_names)
    with nlp.select_pipes(disable=to_disable):
        tokens: list[TokenType] = []
        dependencies: list[TokenType] = []
        entities: list[EntityType] = []

        texts = [sentence.text for sentence in params.sentences]
        for doc, sent in zip(
            nlp.pipe(texts, batch_size=config.spacy_batch_size),
            params.sentences,
        ):
            # map the token index in this Doc to the token index in the list of result tokens
            index_lookup: dict[int, int] = {}

            for token in doc:
                offset: int = sent.offset
                begin = offset + token.idx
                end = offset + token.idx + len(token.text)
                tokens.append(
                    TokenType(
                        begin=begin,
                        end=end,
                        lemma=token.lemma_,
                        pos_value=token.tag_,
                        pos_coarse=token.pos_,
                        morph_value=str(token.morph) if token.has_morph() else None,
                        morph_features=token.morph.to_dict()
                        if token.has_morph()
                        else None,
                    )
                )
                index_lookup[token.i] = len(tokens) - 1

            if nlp.has_pipe("parser"):
                for token in (
                    token
                    for token in doc
                    if not token.is_space and not token.head.is_space
                ):
                    dependent_index = index_lookup[token.i]
                    dependent = tokens[dependent_index]

                    govenor_index = index_lookup[token.head.i]

                    dependencies.append(
                        DependencyType(
                            begin=dependent.begin,
                            end=dependent.end,
                            dependency_type=token.dep_.upper(),
                            governor_index=govenor_index,
                            dependent_index=dependent_index,
                        )
                    )

            if nlp.has_pipe("ner"):
                for entity in doc.ents:
                    entities.append(
                        EntityType(
                            begin=entity.start_char + sent.offset,
                            end=entity.end_char + sent.offset,
                            value=entity.label_,
                            identifier=entity.kb_id_ if entity.kb_id_ else None,
                        )
                    )

        return DuuiResponse(
            metadata=AnnotationMeta(
                name=SETTINGS.component_name,
                version=SETTINGS.component_version,
                spacy_version=spacy.__version__,
                model_lang=nlp.lang,
                model_name=nlp.meta["name"],
                model_pipes=nlp.pipe_names,
                model_spacy_git_version=nlp.meta["spacy_git_version"],
                model_spacy_version=nlp.meta["spacy_version"],
                model_version=nlp.meta["version"],
            ),
            tokens=tokens,
            dependencies=dependencies,
            entities=entities,
        )


###


@app.post(
    "/eos",
    description="End-of-Sentence Detection Endpoint",
)
async def post_eos(
    params: EosRequest,
    request: Request,
) -> EosResponse:
    config: SpacySettings = params.config or SETTINGS

    nlp: Language = get_spacy_model(request.app.state, config)

    logger.info(nlp.pipe_names)
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
            detail="spaCy model does not have a sentence segmentation component",
        )

    with nlp.select_pipes(enable=eos_pipe):
        doc = nlp(params.text)

        sentences = [
            AnnotationType(
                begin=sent.start_char,
                end=sent.end_char,
            )
            for sent in doc.sents
        ]

        return EosResponse(
            metadata=AnnotationMeta(
                name=SETTINGS.component_name + "/eos",
                version=SETTINGS.component_version,
                spacy_version=spacy.__version__,
                model_lang=nlp.lang,
                model_name=nlp.meta["name"],
                model_pipes=nlp.pipe_names,
                model_spacy_git_version=nlp.meta["spacy_git_version"],
                model_spacy_version=nlp.meta["spacy_version"],
                model_version=nlp.meta["version"],
            ),
            sentences=sentences,
        )
