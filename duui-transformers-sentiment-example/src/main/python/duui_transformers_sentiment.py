import logging
from functools import lru_cache
from itertools import chain
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time
from typing import Dict, Union
from datetime import datetime

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
import torch
from transformers import pipeline, __version__ as transformers_version

from .duui.reqres import DUUIResponse, DUUIRequest
from .duui.sentiment import SentimentSentence, SentimentSelection
from .duui.service import Settings, DUUIDocumentation, DUUICapability
from .duui.uima import *
from .models.cardiffnlp_twitter_xlm_roberta_base_sentiment import SUPPORTED_MODEL as CARDIFFNLP_TXRBS

SUPPORTED_MODELS = {
    **CARDIFFNLP_TXRBS,
}

settings = Settings()
supported_languages = sorted(list(set(chain(*[m["languages"] for m in SUPPORTED_MODELS.values()]))))
lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
model_lock = Lock()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab DUUI Transformers Sentiment")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

device = -1
logger.info(f'Using device: {device}')

typesystem_filename = 'src/main/resources/TypeSystemSentiment.xml'
logger.info("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

lua_communication_script_filename = "src/main/lua/duui_transformers_sentiment.lua"
logger.info("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script)

app = FastAPI(
    title=settings.annotator_name,
    description="Transformers-based sentiment analysis for TTLab DUUI",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "baumartz@em.uni-frankfurt.de",
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
def get_documentation() -> DUUIDocumentation:
    capabilities = DUUICapability(
        supported_languages=supported_languages,
        reproducible=True
    )

    documentation = DUUIDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "transformers_version": transformers_version,
            "torch_version": torch.__version__,
        },
        docker_container_id="[TODO]",
        parameters={
            "model_name": SUPPORTED_MODELS,
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
def post_process(request: DUUIRequest) -> DUUIResponse:
    processed_selections = []
    meta = None
    modification_meta = None

    dt = datetime.now()
    print(dt, 'Started processing', flush=True)

    try:
        modification_timestamp_seconds = int(time())

        logger.debug("Received:")
        logger.debug(request)

        if request.model_name not in SUPPORTED_MODELS:
            raise Exception(f"Model \"{request.model_name}\" is not supported!")

        if request.lang not in SUPPORTED_MODELS[request.model_name]["languages"]:
            raise Exception(f"Document language \"{request.lang}\" is not supported by model \"{request.model_name}\"!")

        logger.info("Using model: \"%s\"", request.model_name)
        model_data = SUPPORTED_MODELS[request.model_name]
        logger.debug(model_data)

        for selection in request.selections:
            processed_sentences = process_selection(request.model_name, model_data, selection, request.doc_len, request.batch_size, request.ignore_max_length_truncation_padding)

            processed_selections.append(
                SentimentSelection(
                    selection=selection.selection,
                    sentences=processed_sentences
                )
            )

        meta = UimaAnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=request.model_name,
            modelVersion=model_data["version"],
        )

        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version})"
        modification_meta = UimaDocumentModification(
            user="DUUI",
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )

    except Exception as ex:
        logger.exception(ex)

    dte = datetime.now()
    print(dte, 'Finished processing', flush=True)
    print('Time elapsed', f'{dte-dt}', flush=True)

    return DUUIResponse(
        selections=processed_selections,
        meta=meta,
        modification_meta=modification_meta
    )


@lru_cache_with_size
def load_model(model_name, model_version, labels_count):
    return pipeline(
        "sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        revision=model_version,
        top_k=labels_count,
        device=device
    )


def map_sentiment(sentiment_result: List[Dict[str, Union[str, float]]], sentiment_mapping: Dict[str, float], sentiment_polarity: Dict[str, List[str]], sentence: UimaSentence) -> SentimentSentence:
    # get label from top result and map to sentiment values -1, 0 or 1
    sentiment_value = 0.0
    top_result = sentiment_result[0]
    if top_result["label"] in sentiment_mapping:
        sentiment_value = sentiment_mapping[top_result["label"]]

    # get scores of all labels
    details = {
        s["label"]: s["score"]
        for s in sentiment_result
    }

    # calculate polarity: pos-neg
    polarities = {
        "pos": 0,
        "neu": 0,
        "neg": 0
    }
    for p in polarities:
        for l in sentiment_polarity[p]:
            for s in sentiment_result:
                if s["label"] == l:
                    polarities[p] += s["score"]

    polarity = polarities["pos"] - polarities["neg"]

    return SentimentSentence(
        sentence=sentence,
        sentiment=sentiment_value,
        score=top_result["score"],
        details=details,
        polarity=polarity,
        **polarities
    )


def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    # NOTE this should only be used in tools that to not need to return begin or end indices, check the duui-spacy
    #  tool for an example in this case
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text


def process_selection(model_name, model_data, selection, doc_len, batch_size, ignore_max_length_truncation_padding):
    for s in selection.sentences:
        s.text = fix_unicode_problems(s.text)

    texts = [
        model_data["preprocess"](s.text)
        for s in selection.sentences
    ]
    logger.debug("Preprocessed texts:")
    logger.debug(texts)

    with model_lock:
        sentiment_analysis = load_model(model_name, model_data["version"], len(model_data["mapping"]))

        if ignore_max_length_truncation_padding:
            results = sentiment_analysis(
                texts, batch_size=batch_size
            )
        else:
            results = sentiment_analysis(
                texts, truncation=True, padding=True, max_length=model_data["max_length"], batch_size=batch_size
            )

    processed_sentences = [
        map_sentiment(r, model_data["mapping"], model_data["3sentiment"], s)
        for s, r
        in zip(selection.sentences, results)
    ]

    if len(results) > 1:
        begin = 0
        end = doc_len

        sentiments = 0
        for sentence in processed_sentences:
            sentiments += sentence.sentiment
        sentiment = sentiments / len(processed_sentences)

        scores = 0
        for sentence in processed_sentences:
            scores += sentence.score
        score = scores / len(processed_sentences)

        details = {}
        for sentence in processed_sentences:
            for d in sentence.details:
                if d not in details:
                    details[d] = 0
                details[d] += sentence.details[d]
        for d in details:
            details[d] = details[d] / len(processed_sentences)

        polaritys = 0
        for sentence in processed_sentences:
            polaritys += sentence.polarity
        polarity = polaritys / len(processed_sentences)

        poss = 0
        for sentence in processed_sentences:
            poss += sentence.pos
        pos = poss / len(processed_sentences)

        neus = 0
        for sentence in processed_sentences:
            neus += sentence.neu
        neu = neus / len(processed_sentences)

        negs = 0
        for sentence in processed_sentences:
            negs += sentence.neg
        neg = negs / len(processed_sentences)

        processed_sentences.append(
            SentimentSentence(
                sentence=UimaSentence(
                    text="",
                    begin=begin,
                    end=end,
                ),
                sentiment=sentiment,
                score=score,
                details=details,
                polarity=polarity,
                pos=pos,
                neu=neu,
                neg=neg
            )
        )

    return processed_sentences
