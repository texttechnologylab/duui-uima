from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from typing import List, Optional
from time import time

from cassis import *
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse
from trankit import Pipeline, __version__ as trankit_version


# TODO
DUUI_DEFAULT_LANGUAGE = "de"


SUPPORTED_LANGS = {
    "en",
    "de",
}

LANGUAGE_MAPPING = {
    "en": "english",
    "de": "german",
}


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class Sentence(BaseModel):
    begin: int
    end: int


class DUUIRequest(BaseModel):
    doc_text: str
    lang: str


class DUUIResponse(BaseModel):
    sentences: Optional[List[Sentence]]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: str


class Settings(BaseSettings):
    model_name: str
    cuda: int
    annotator_name: str
    annotator_version: str

    class Config:
        env_prefix = 'duui_sentencizer_trankit_'


settings = Settings()
lru_cache_with_size = lru_cache(maxsize=3)


@lru_cache_with_size
def load_pipeline(lang, embedding, gpu) -> Pipeline:
    use_gpu = bool(gpu)
    print("Loading pipeline for", lang, "with", embedding, "embedding and gpu", gpu)
    return Pipeline(lang, embedding=embedding, gpu=use_gpu)


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="TRANKIT",
    description="Trankit (spacy alternative) Implementation for TTLab TextImager DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "leon.hammerla@gmx.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")

typesystem_filename = 'TypeSystem.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")


@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    json_item = {
        "inputs": [""],
        "outputs": ["de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication


@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "stanza_version": trankit_version,
        },
    )
    return documentation


@app.post("/v1/process")
def post_process(request: DUUIRequest) -> DUUIResponse:
    modification_timestamp_seconds = int(time())

    # TODO
    lang = request.lang
    if lang not in SUPPORTED_LANGS:
        print("WARNING: Unsupported language detected:", lang, "using default language:", DUUI_DEFAULT_LANGUAGE)
        lang = DUUI_DEFAULT_LANGUAGE

    # TODO allow usage of "auto" with parameter
    lang = LANGUAGE_MAPPING.get(lang, "auto")
    print("Language detected:", lang, "parsed from", request.lang)

    pipeline = load_pipeline(lang, embedding=settings.model_name, gpu=settings.cuda)

    sents = []
    if len(request.doc_text.strip()) > 0:
        res = pipeline.ssplit(request.doc_text)
        for sent in res["sentences"]:
            sents.append(Sentence(
                begin=sent["dspan"][0],
                end=sent["dspan"][1],
            ))
    else:
        print("Warning: Empty input text detected.")

    meta = AnnotationMeta(
        name=settings.annotator_name,
        version=settings.annotator_version,
        modelName="Trankit",
        modelVersion=trankit_version
    )

    modification_meta = DocumentModification(
        user=settings.annotator_name,
        timestamp=modification_timestamp_seconds,
        comment=f"{settings.annotator_name} ({settings.annotator_version}), Trankit ({trankit_version})"
    )

    duration = int(time()) - modification_timestamp_seconds
    print("Processed in", duration, "seconds")

    return DUUIResponse(
        sentences=sents,
        meta=meta,
        modification_meta=modification_meta,
    )
