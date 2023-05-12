import logging
import math
import os
import sys
from functools import lru_cache
from typing import Final, Dict, List, Optional, Iterable, Callable, TypeVar

import flair
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from flair.data import Sentence
from flair.models import SequenceTagger
from pydantic import BaseModel

T = TypeVar("T")

logger = logging.getLogger("fastapi")

MODEL_CACHE_SIZE: Final[int] = int(os.environ.get("MODEL_CACHE_SIZE", 1))
logger.info(f"MODEL_CACHE_SIZE={MODEL_CACHE_SIZE}")
BATCH_SIZE: Final[int] = int(os.environ.get("FLAIR_BATCH_SIZE", 128))
logger.info(f"BATCH_SIZE={BATCH_SIZE}")

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="Flair NER tagger",
    description="Flair NER Tagger for the TTLab TextImager DUUI",
    version=os.environ.get("FLAIR_NER_VERSION", "0.0.1"),
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "manuel.stoeckel@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

logger.debug("Loading Lua communication layer from file")
with open("communication_layer.lua", "r", encoding="utf-8") as f:
    lua_communication_script = f.read()

logger.debug("Loading type system from file")
with open("dkpro-core-types.xml", "rb") as f:
    type_system = f.read()

lang_code_to_model_map: Final[Dict[str, str]] = {
    # NER (4-class) 	English 	Conll-03 	93.03 (F1)
    "en": "flair/ner-english",
    # NER (4-class) 	English 	Conll-03 	93.03 (F1)
    "en-ner": "flair/ner-english",
    # NER (4-class) 	English 	Conll-03 	92.75 (F1) 	(fast model)
    "en-fast": "flair/ner-english-fast",
    # NER (4-class) 	English / Multilingual 	Conll-03 	94.09 (F1) 	(large model)
    "en-large": "flair/ner-english-large",
    # NER (4-class) 	English 	Conll-03 	93.24 (F1) 	(memory inefficient)
    "en-pooled": "ner-pooled",
    # NER (18-class) 	English 	Ontonotes 	89.06 (F1)
    "en-ontonotes": "flair/ner-english-ontonotes",
    # NER (18-class) 	English 	Ontonotes 	89.27 (F1) 	(fast model)
    "en-ontonotes-fast": "flair/ner-english-ontonotes-fast",
    # NER (18-class) 	English / Multilingual 	Ontonotes 	90.93 (F1) 	(large model)
    "en-ontonotes-large": "flair/ner-english-ontonotes-large",
    # NER (4-class) 	Arabic 	AQMAR & ANERcorp (curated) 	86.66 (F1)
    "ar": "ar-ner",
    # NER (4-class) 	Arabic 	AQMAR & ANERcorp (curated) 	86.66 (F1)
    "ar-ner": "ar-ner",
    # NER (4-class) 	Danish 	Danish NER dataset 		AmaliePauli
    "da": "flair/ner-danish",
    # NER (4-class) 	Danish 	Danish NER dataset 		AmaliePauli
    "da-ner": "flair/ner-danish",
    # NER (4-class) 	German 	Conll-03 	87.94 (F1)
    "de": "flair/ner-german",
    # NER (4-class) 	German 	Conll-03 	87.94 (F1)
    "de-ner": "flair/ner-german",
    # NER (4-class) 	German / Multilingual 	Conll-03 	92.31 (F1)
    "de-large": "flair/ner-german-large",
    # NER (4-class) 	German 	Germeval 	84.90 (F1)
    "de-germeval": "de-ner-germeval",
    # NER (legal text) 	German 	LER dataset 	96.35 (F1)
    "de-legal": "flair/ner-german-legal",
    # NER (4-class) 	French 	WikiNER (aij-wikiner-fr-wp3) 	95.57 (F1) 	mhham
    "fr": "flair/ner-french",
    # NER (4-class) 	French 	WikiNER (aij-wikiner-fr-wp3) 	95.57 (F1) 	mhham
    "fr-ner": "flair/ner-french",
    # NER (4-class) 	Spanish 	CoNLL-03 	90.54 (F1) 	mhham
    "es": "flair/ner-spanish-large",
    # NER (4-class) 	Spanish 	CoNLL-03 	90.54 (F1) 	mhham
    "es-ner": "flair/ner-spanish-large",
    "nl": "flair/ner-dutch",  # NER (4-class) 	Dutch 	CoNLL 2002 	92.58 (F1)
    # NER (4-class) 	Dutch 	CoNLL 2002 	92.58 (F1)
    "nl-ner": "flair/ner-dutch",
    # NER (4-class) 	Dutch 	Conll-03 	95.25 (F1)
    "nl-large": "flair/ner-dutch-large",
    # NER (4-class) 	Dutch 	CoNLL 2002 	90.79 (F1)
    "nl-rnn": "nl-ner-rnn",
    # NER (4-class) 	Ukrainian 	NER-UK dataset 	86.05 (F1) 	dchaplinsky
    "uk": "dchaplinsky/flair-uk-ner",
    # NER (4-class) 	Ukrainian 	NER-UK dataset 	86.05 (F1) 	dchaplinsky
    "uk-ner": "dchaplinsky/flair-uk-ner",
}
supported_languages: Final[List[str]] = list(
    sorted(lang_code_to_model_map.keys()))


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(content=type_system, media_type="application/xml")


# Capabilities
class TextImagerCapability(BaseModel):
    # List of supported languages by the annotator
    # TODO how to handle language?
    # - ISO 639-1 (two letter codes) as default in meta data
    # - ISO 639-3 (three letters) optionally in extra meta to allow a finer mapping
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
    implementation_lang: Optional[str]
    # Optional map of additional meta data
    meta: Optional[dict]
    # Docker container id, if any
    docker_container_id: Optional[str]
    # Optional map of supported parameters
    parameters: Optional[dict]
    # Capabilities of this annotator
    capability: TextImagerCapability
    # Analysis engine XML, if available
    implementation_specific: Optional[str]


ner_types: Final[Dict[str, str]] = {
    "Animal": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Animal",
    "Cardinal": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Cardinal",
    "ContactInfo": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.ContactInfo",
    "Date": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Date",
    "Disease": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Disease",
    "Event": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Event",
    "Fac": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Fac",
    "FacDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.FacDesc",
    "Game": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Game",
    "Gpe": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Gpe",
    "GpeDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.GpeDesc",
    "Language": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Language",
    "Law": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Law",
    "Location": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location",
    "Money": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Money",
    "NamedEntity": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity",
    "Nationality": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Nationality",
    "Norp": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Norp",
    "Ordinal": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Ordinal",
    "OrgDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.OrgDesc",
    "Organization": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Organization",
    "PerDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.PerDesc",
    "Percent": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Percent",
    "Person": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Person",
    "Plant": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Plant",
    "Product": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Product",
    "ProductDesc": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.ProductDesc",
    "Quantity": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Quantity",
    "Substance": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Substance",
    "Time": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Time",
    "WorkOfArt": "de.tudarmstadt.ukp.dkpro.core.api.ner.type.WorkOfArt",
}
ner_base_type: Final[str] = ner_types["NamedEntity"]
ner_tag_map: Final[Dict[str, str]] = {
    "LOC": "Location",
    "ORG": "Organization",
    "PER": "Person",
}


class DkproNer(BaseModel):
    """
    Models the DKPRO NamedEntity type, de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity
    """

    # Inherited from uima.tcas.Annotation
    begin: int
    # Inherited from uima.tcas.Annotation
    end: int
    # The class/category of the named entity, e.g. person, location, etc.
    value: str
    # Identifier of the named entity, e.g. a reference into a person database.
    identifier: Optional[str]
    # The fully *.ner.type.NamedEntity subclass.
    ner_type: str = ner_base_type


def get_ner_type(o_tag: str) -> str:
    if o_tag in ner_tag_map:
        return ner_types[ner_tag_map[o_tag]]

    tag = "".join(map(str.title, o_tag.split("_")))

    return ner_types.get(tag, ner_base_type)


class DkproSentence(BaseModel):
    """
    Models the DKPRO Sentence type, de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence
    """

    offset: int
    coveredText: str


# Return documentation info
@app.get("/v1/documentation")
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=supported_languages, reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name="Flair POS - DUII",
        version="0.0.1",
        implementation_lang="Python",
        meta={"python_version": sys.version,
              "flair_version": flair.__version__},
        docker_container_id="docker.texttechnologylab.org/flair/pos:latest",
        parameters={
            "language": "de",
        },
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


class Message(BaseModel):
    message: str


class TextImagerRequest(BaseModel):
    text: str
    language: str
    sentences: List[DkproSentence]
    optional_tag_map: Optional[Dict[str, str]]


class TextImagerResponse(BaseModel):
    tags: List[DkproNer]


@lru_cache(maxsize=MODEL_CACHE_SIZE)
def load_model(lang: str) -> SequenceTagger:
    return SequenceTagger.load(lang)


def batcher(iterable: List[T], batch_size=BATCH_SIZE) -> Iterable[List[T]]:
    _len = len(iterable)
    for start in range(0, _len, batch_size):
        yield list(iterable[start:start + batch_size])


def flatten(iterable: Iterable[Iterable[T]]) -> Iterable[T]:
    for it in iterable:
        yield from it


@app.post(
    "/v1/process",
    response_model=TextImagerResponse,
    responses={
        400: {
            "model": Message,
            "description": "There was an error with the request",
        },
    },
)
def post_process(request: TextImagerRequest):
    language = request.language
    if language not in supported_languages:
        supported_lang_string = ", ".join(supported_languages)
        return JSONResponse(
            status_code=400,
            content={
                "message": (
                    f"The selected language '{language}' is not supported. "
                    f"Supported languages: {supported_lang_string}"
                )
            },
        )
    model = load_model(lang_code_to_model_map[language])
    text = request.text
    if request.optional_tag_map:
        tag_map = request.optional_tag_map

        def tag_lookup(key):
            return tag_map.get(key, get_ner_type(key))

    else:
        tag_lookup = get_ner_type

    if request.sentences:
        total_batches = math.ceil(len(request.sentences) / BATCH_SIZE * 1.)

        def process_verbose(idx, batch: List[T]) -> Iterable[T]:
            logger.info(f"Processing batch {idx}/{total_batches}")
            return process_batch(model, text, batch, tag_lookup)

        tags = list(flatten(
            process_verbose(idx, batch)
            for idx, batch in enumerate(batcher(request.sentences, BATCH_SIZE), start=1)
        ))
        return TextImagerResponse(tags=tags)
    else:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"The input MUST be sentence segmented for {app.title} to work."
            },
        )


def process_batch(
        model: SequenceTagger,
        text: str,
        batch: List[DkproSentence],
        tag_lookup: Callable[[str], str]
) -> Iterable[DkproNer]:
    sentences: List[Sentence] = [
        Sentence(
            dkpro_sentence.coveredText,
            start_position=dkpro_sentence.offset,
        )
        for dkpro_sentence in batch
    ]
    model.predict(sentences)
    for sentence in sentences:
        for label in sentence.get_labels("ner"):
            begin = label.data_point.start_position + sentence.start_position
            end = label.data_point.end_position + sentence.start_position
            tag_type = tag_lookup(label.value)
            yield DkproNer(
                begin=begin,
                end=end,
                value=label.value,
                identifier=None,
                ner_type=tag_type,
            )
