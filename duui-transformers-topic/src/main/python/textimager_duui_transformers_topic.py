import logging
from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import Dict, Union

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
import torch
from transformers import pipeline, __version__ as transformers_version

from .duui.reqres import TextImagerResponse, TextImagerRequest
from .duui.topic import TopicSentence, TopicSelection
from .duui.service import Settings, TextImagerDocumentation, TextImagerCapability
from .duui.uima import *


settings = Settings()
supported_languages = ["DE"]
lru_cache_with_size = lru_cache(maxsize=settings.textimager_duui_transformers_topic_model_cache_size)
model_lock = Lock()

logging.basicConfig(level=settings.textimager_duui_transformers_topic_log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Transformers Topic")
logger.info("Name: %s", settings.textimager_duui_transformers_topic_annotator_name)
logger.info("Version: %s", settings.textimager_duui_transformers_topic_annotator_version)

device = 0 if torch.cuda.is_available() else -1
logger.info(f'USING {device}')

typesystem_filename = 'src/main/resources/TypeSystemTopic.xml'
logger.info("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

lua_communication_script_filename = "src/main/lua/textimager_duui_transformers_topic.lua"
logger.info("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script)

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.textimager_duui_transformers_topic_annotator_name,
    description="Transformers-based topic analysis for TTLab TextImager DUUI",
    version=settings.textimager_duui_transformers_topic_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "bagci@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=supported_languages,
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.textimager_duui_transformers_topic_annotator_name,
        version=settings.textimager_duui_transformers_topic_annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
        },
        docker_container_id="[TODO]",
        parameters={
            "model_name": ["__huggingface__"],
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    processed_selections = []
    meta = None
    modification_meta = None

    clean_cuda_cache()

    try:
        modification_timestamp_seconds = int(time())

        logger.debug("Received:")
        logger.debug(request)

        logger.info("Using model: \"%s\"", request.model_name)
        mv = ""
        for selection in request.selections:
            processed_sentences, model_version = process_selection(request.model_name, selection, request.doc_len)
            mv = model_version
            processed_selections.append(
                TopicSelection(
                    selection=selection.selection,
                    sentences=processed_sentences
                )
            )

        meta = UimaAnnotationMeta(
            name=settings.textimager_duui_transformers_topic_annotator_name,
            version=settings.textimager_duui_transformers_topic_annotator_version,
            modelName=request.model_name,
            modelVersion=f"{mv}",
        )

        modification_meta_comment = f"{settings.textimager_duui_transformers_topic_annotator_name} ({settings.textimager_duui_transformers_topic_annotator_version})"
        modification_meta = UimaDocumentModification(
            user="TextImager",
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )

    except Exception as ex:
        logger.exception(ex)

    logger.debug(processed_selections)

    clean_cuda_cache()

    return TextImagerResponse(
        selections=processed_selections,
        meta=meta,
        modification_meta=modification_meta
    )


@lru_cache_with_size
def load_model(model_name):

    return pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        device=device,
    )


def map_topic(topic_result: List[Dict[str, Union[str, float]]], sentence: UimaSentence) -> TopicSentence:
    return TopicSentence(
        sentence=sentence,
        topics=topic_result
    )


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def process_selection(model_name, selection, doc_len):
    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    texts = [
        s.text
        for s in selection.sentences
    ]
    logger.debug("Preprocessed texts:")
    logger.debug(texts)

    with model_lock:
        classifier = load_model(model_name)

        results = classifier(texts, batch_size=128)

    processed_sentences = [
        map_topic(r, s)
        for s, r
        in zip(selection.sentences, results)
    ]

    return processed_sentences, classifier.model._version


def clean_cuda_cache():
    if device >= 0:
        logger.info('emptying cuda cache')
        torch.cuda.empty_cache()
        logger.info('cuda cache empty')
