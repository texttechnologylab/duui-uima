import os
import logging
import uvicorn
import torch

from functools import lru_cache
from platform import python_version
from sys import version as sys_version
from threading import Lock
from time import time

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from torch import __version__ as torch_version
from transformers import __version__ as transformers_version
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from duui.reqres import TextImagerResponse, TextImagerRequest
from duui.service import Settings, TextImagerDocumentation, TextImagerCapability
from duui.uima import *

#os.environ["TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_MODEL_CACHE_SIZE"] = "1"
#os.environ["TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_LOG_LEVEL"] = "DEBUG"
#os.environ["TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_NAME"] = "textimager-duui-transformers-summary"
#os.environ["TEXTIMAGER_DUUI_TRANSFORMERS_SUMMARY_ANNOTATOR_VERSION"] = "0.0.1"


settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.textimager_duui_transformers_summary_model_cache_size)
model_lock = Lock()

SUPPORTED_MODELS = ["Google T5-base", "Google Pegasus", "Facebook Bart-large", "Google MT5-small", "Facebook DistilBART"]
supported_languages = ["de", "ger"]


logging.basicConfig(level=settings.textimager_duui_transformers_summary_log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI Transformers Summary")
logger.info("Name: %s", settings.textimager_duui_transformers_summary_annotator_name)
logger.info("Version: %s", settings.textimager_duui_transformers_summary_annotator_version)

typesystem_filename = 'TypeSystemSentiment.xml'
logger.info("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

lua_communication_script_filename = "textimager_duui_transformers_summary.lua"
logger.info("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script)

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.textimager_duui_transformers_summary_annotator_name,
    description="Transformers-based summarizer for TTLab TextImager DUUI",
    version=settings.textimager_duui_transformers_summary_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "henlein@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)
Loaded_Model=None

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
        annotator_name=settings.textimager_duui_transformers_summary_annotator_name,
        version=settings.textimager_duui_transformers_summary_annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
            "transformers_version": transformers_version,
            "torch_version": torch_version,
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

def load_model(model_name):
    prefix = ""
    tokenizer, model = None, None

    if model_name == "Google T5-base":
        tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")
        model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/T5-Base_GNAD")
        prefix = "summarize: "
    elif model_name == "Google Pegasus":
        tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/PegasusXSUM_GNAD")
        model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/PegasusXSUM_GNAD")
    elif model_name == "Facebook Bart-large":
        tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/BART_large_CNN_GNAD")
        model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/BART_large_CNN_GNAD")
    elif model_name == "Google MT5-small":
        tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/MT5_small_sum-de_GNAD")
        model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/MT5_small_sum-de_GNAD")
        prefix = "summarize: "
    elif model_name == "Facebook DistilBART":
        tokenizer = AutoTokenizer.from_pretrained("Einmalumdiewelt/DistilBART_CNN_GNAD")
        model = AutoModelForSeq2SeqLM.from_pretrained("Einmalumdiewelt/DistilBART_CNN_GNAD")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == 'cuda':
        torch.cuda.empty_cache() # clear GPU memory

    model.to(device)



    return tokenizer, model, prefix, device


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    global Loaded_Model
    print(request)
    meta = None
    modification_meta = None

    generated_summaries = []
    try:
        modification_timestamp_seconds = int(time())

        logger.debug("Received:")
        logger.debug(request)

        if request.model_name not in SUPPORTED_MODELS:
            raise Exception(f"Model \"{request.model_name}\" is not supported!")

        if request.lang not in supported_languages:
            raise Exception(f"Document language \"{request.lang}\" is currently not supported!")

        if request.summary_length < 25:
            raise Exception(f"Only summary_length >= 25 is supported!")

        logger.info("Using model: \"%s\"", request.model_name)

        if Loaded_Model != request.model_name:
            tokenizer, model, prefix, device = load_model(request.model_name)
            Loaded_Model = request.model_name

        meta = UimaAnnotationMeta(
            name=settings.textimager_duui_transformers_summary_annotator_name,
            version=settings.textimager_duui_transformers_summary_annotator_version,
            modelName=request.model_name,
            modelVersion="1.0.0",
        )

        modification_meta_comment = f"{settings.textimager_duui_transformers_summary_annotator_name} ({settings.textimager_duui_transformers_summary_annotator_version})"
        modification_meta = UimaDocumentModification(
            user="TextImager",
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )

        if request.docs[0] == "#leer#":
            return TextImagerResponse(
                summaries=[""],
                meta=meta,
                modification_meta=modification_meta
            )

        for text in request.docs:

            # https://huggingface.co/spaces/Einmalumdiewelt/German_text_summarization/blob/main/app.py
            inputs = tokenizer(
                prefix + text,
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors='pt').to(device)

            if request.summary_length == 25:
                # make sure models actually generate something
                preds = model.generate(**inputs, max_length=request.summary_length + 5, min_length=request.summary_length - 20)
            else:
                preds = model.generate(**inputs, max_length=request.summary_length + 25, min_length=request.summary_length - 25)
            # we decode the predictions to store them
            decoded_predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
            generated_summaries.append(decoded_predictions[0])

    except Exception as ex:
        logger.exception(ex)


    return TextImagerResponse(
        summaries=generated_summaries,
        meta=meta,
        modification_meta=modification_meta
    )


if __name__ == "__main__":
    uvicorn.run(app)
