import importlib
import json
import logging
from platform import python_version
from sys import version as sys_version
from time import time
from typing import List, Optional

from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from langchain_core.messages.base import message_to_dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_core_llm_rating_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI CORE LLM Rating")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES = [
    "org.texttechnologylab.type.llm.prompt.Result"
]

TEXTIMAGER_ANNOTATOR_INPUT_TYPES = [
    "org.texttechnologylab.type.llm.prompt.Prompt",
    "org.texttechnologylab.type.llm.prompt.Message"
]

SUPPORTED_LANGS = [
    # all
]


class LLMMessage(BaseModel):
    role: str = None
    content: str
    class_module: str = None
    class_name: str = None
    fillable: bool = False
    context_name: str = None
    ref: int   # internal cas annotation id


class LLMPrompt(BaseModel):
    messages: List[LLMMessage]
    args: str  # json string
    ref: int   # internal cas annotation id


class LLMResult(BaseModel):
    meta: str  # json string
    prompt_ref: int   # internal cas annotation id
    message_ref: int   # internal cas annotation id


class TextImagerRequest(BaseModel):
    prompts: List[LLMPrompt]
    llm_args: str  # json string


class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str


class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


class TextImagerResponse(BaseModel):
    llm_results: List[LLMResult]
    prompts: List[LLMPrompt]
    meta: Optional[AnnotationMeta]
    modification_meta: Optional[DocumentModification]


class TextImagerCapability(BaseModel):
    supported_languages: List[str]
    reproducible: bool


class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str]
    meta: Optional[dict]
    docker_container_id: Optional[str]
    parameters: Optional[dict]
    capability: TextImagerCapability
    implementation_specific: Optional[str]


class TextImagerInputOutput(BaseModel):
    inputs: List[str]
    outputs: List[str]


typesystem_filename = 'src/main/resources/TypeSystem.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    typesystem_xml_content = typesystem.to_xml().encode("utf-8")
    logger.debug("Base typesystem:")
    logger.debug(typesystem_xml_content)

lua_communication_script_filename = "src/main/lua/communication.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)
with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
    logger.debug("Lua communication script:")
    logger.debug(lua_communication_script_filename)

app = FastAPI(
    title=settings.annotator_name,
    description="TTLab TextImager DUUI CORE LLM Rating",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team - Daniel Baumartz",
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
def get_documentation() -> TextImagerDocumentation:
    capabilities = TextImagerCapability(
        supported_languages=SUPPORTED_LANGS,
        reproducible=True
    )

    documentation = TextImagerDocumentation(
        annotator_name=settings.annotator_name,
        version=settings.annotator_version,
        implementation_lang="Python",
        meta={
            "python_version": python_version(),
            "python_version_full": sys_version,
        },
        docker_container_id="[TODO]",
        parameters={},
        capability=capabilities,
        implementation_specific=None,
    )

    return documentation


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem_xml_content,
        media_type="application/xml"
    )


@app.get("/v1/details/input_output")
def get_input_output() -> TextImagerInputOutput:
    return TextImagerInputOutput(
        inputs=TEXTIMAGER_ANNOTATOR_INPUT_TYPES,
        outputs=TEXTIMAGER_ANNOTATOR_OUTPUT_TYPES
    )


def _query_llm(prompt_messages, llm, prompt_args):
    prompt_messages = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt_messages | llm

    llm_result = chain.invoke(prompt_args)
    llm_result = message_to_dict(llm_result)
    return llm_result


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    llm_results = []
    meta = None
    modification_meta = None

    try:
        llm_args = json.loads(request.llm_args)
        llm = ChatOllama(**llm_args)

        for prompt in request.prompts:
            # context data that is given to llm invocation
            # consists of "global" prompt arguments and additionally extracted content from messages
            context = json.loads(prompt.args)

            prompt_messages = []
            for message in prompt.messages:
                # check if this message should be filled by the model
                # only fill if no content (== json encoded empty string) is available
                if message.fillable is True and message.content == "\"\"":
                    llm_t_start = time()
                    llm_result = _query_llm(prompt_messages, llm, context)
                    llm_t_end = time()
                    llm_t_duration = llm_t_end - llm_t_start
                    llm_content = llm_result["data"]["content"]

                    # add the result to this message
                    # NOTE that we always encode the content as json, as we expect for all "Langchain" messages
                    message.content = json.dumps(llm_content)

                    # add results to output
                    del llm_result["data"]["content"]
                    llm_result = {
                        **llm_result,
                        "prompt_args": context,
                        "llm_args": llm_args,
                        "llm_t_start": llm_t_start,
                        "llm_t_end": llm_t_end,
                        "llm_t_duration": llm_t_duration
                    }
                    llm_results.append(LLMResult(
                        meta=json.dumps(llm_result),
                        prompt_ref=prompt.ref,
                        message_ref=message.ref
                    ))

                # if set, extract this messages content into the context
                if message.context_name is not None:
                    if message.context_name in context:
                        logger.warning("Context name \"%s\" already exists, overwriting", message.context_name)
                    # content is always json encoded on Langchain-class-messages
                    context[message.context_name] = json.loads(message.content)

                # add message to prompt
                # we support three types of messages:
                # - simple tuple consisting of content and role, this supports placeholders
                # - message-class based, consisting of content (and implicit role), this does not support placeholder resolution
                # - template-class based, consisting of content with support for placeholders
                if message.class_module is not None and message.class_name is not None:
                    # create Langchain-class-based message
                    module = importlib.import_module(message.class_module)
                    constructor = getattr(module, message.class_name)

                    # content is always json encoded on Langchain-class-messages
                    msg_content = json.loads(message.content)

                    if "prompt" in constructor.model_fields:
                        prompt_messages.append(
                            constructor.from_template(msg_content)
                        )
                    elif "content" in constructor.model_fields:
                        prompt_messages.append(
                            constructor(content=msg_content)
                        )
                else:
                    prompt_messages.append((
                        message.role,
                        message.content
                    ))

        try:
            model_name, _, model_version = llm_args["model"].partition(":")
        except:
            model_name = llm_args["model"]
            model_version = ""

        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=model_name,
            modelVersion=model_version
        )

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), {llm_args['model']}"
        )

    except Exception as ex:
        logger.exception(ex)

    logger.debug(meta)
    logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    return TextImagerResponse(
        llm_results=llm_results,
        prompts=request.prompts,
        meta=meta,
        modification_meta=modification_meta,
    )
