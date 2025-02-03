import logging
from base64 import b64decode
from platform import python_version
from sys import version as sys_version
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import time
from typing import List, Optional

import cv2
import numpy as np
import pytesseract
import textract
from PIL import Image
from cassis import load_typesystem
from fastapi import FastAPI, Response
from fastapi.responses import PlainTextResponse
from pdf2image import convert_from_path
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    annotator_name: str
    annotator_version: str
    log_level: str

    class Config:
        env_prefix = 'duui_pdf_extract_ocr_'


settings = Settings()

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
logger.info("TTLab TextImager DUUI PDF Extract/OCR")
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

SUPPORTED_LANGS = {
    "en": "eng",
    "de": "deu",
}


class TextImagerRequest(BaseModel):
    lang: str
    data: str
    min_chars: int = 100
    ocr_dpi: int = 200
    ocr_preprocess: bool = False


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
    text: str
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
    description="TTLab TextImager DUUI PDF Extract/OCR",
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
        supported_languages=sorted(SUPPORTED_LANGS.keys()),
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
        # Input is PDF bytes in SofA
        inputs=[],
        # Output is text for SofA
        outputs=[]
    )


@app.post("/v1/process")
def post_process(request: TextImagerRequest) -> TextImagerResponse:
    modification_timestamp_seconds = int(time())

    text = None
    meta = None
    modification_meta = None
    model_name = None
    model_version = None

    if request.lang not in SUPPORTED_LANGS:
        raise Exception(f"Unsupported language: {request.lang}, supported languages are {', '.join(SUPPORTED_LANGS)}")
    else:
        # convert language to tesseract format
        lang = SUPPORTED_LANGS[request.lang]

    with NamedTemporaryFile(suffix=".pdf") as pdf_temp_file:
        # the pdf is base64 encoded as we can not send bytes directly
        pdf_temp_file.write(b64decode(request.data))

        # first try: extract text directly
        try:
            model_name = "textract"
            model_version = textract.VERSION

            # see https://github.com/texttechnologylab/GerParCor/blob/main/python_Text_extraction/pdf_to_text/pdftext_to_text.py#L14
            temp_text = textract.process(pdf_temp_file.name, method='pdftotext').decode("utf-8")

            if len(temp_text) < request.min_chars:
                logger.warning(f"Text extraction produced less than {request.min_chars} characters")
            else:
                text = temp_text

        except Exception as ex:
            logger.exception(ex)

        # second try: OCR
        if text is None:
            logger.warning(f"Text extraction failed, trying OCR with dpi={request.ocr_dpi} and preprocessing={request.ocr_preprocess} ...")
            try:
                model_name = "pytesseract"
                model_version = pytesseract.__version__

                with TemporaryDirectory() as ocr_temp_dir:
                    # see https://github.com/texttechnologylab/GerParCor/blob/main/python_Text_extraction/pdf_to_text/scanned_pdf_to_text.py#L18
                    images = convert_from_path(pdf_temp_file.name, dpi=request.ocr_dpi, fmt="JPEG", output_folder=ocr_temp_dir)

                    temp_text = ""
                    for image in images:
                        # see https://github.com/texttechnologylab/GerParCor/blob/main/python_Text_extraction/pdf_to_text/scanned_pdf_to_text.py#L53
                        if lang == "frk" or request.ocr_preprocess:
                            preprocess_bad_quality_text(image.filename)
                            image = Image.open(image.filename)

                        # see https://github.com/texttechnologylab/GerParCor/blob/main/python_Text_extraction/pdf_to_text/scanned_pdf_to_text.py#L102
                        temp_text += pytesseract.image_to_string(image.filename, lang=lang)

                    # see https://github.com/texttechnologylab/GerParCor/blob/main/python_Text_extraction/pdf_to_text/scanned_pdf_to_text.py#L104
                    temp_text = temp_text.replace('-\n', '')

                    if len(temp_text) < request.min_chars:
                        logger.warning(f"OCR produced less than {request.min_chars} characters")
                    else:
                        text = temp_text

            except Exception as ex:
                logger.exception(ex)

    if text is None:
        raise Exception("Failed to extract text with both, text and OCR!")
    else:
        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=model_name,
            modelVersion=model_version
        )
        logger.debug(meta)

        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=f"{settings.annotator_name} ({settings.annotator_version}), {model_name} ({model_version})"
        )
        logger.debug(modification_meta)

    duration = int(time()) - modification_timestamp_seconds
    logger.info("Processed in %d seconds", duration)

    return TextImagerResponse(
        text=text,
        meta=meta,
        modification_meta=modification_meta,
    )

# see https://github.com/texttechnologylab/GerParCor/blob/main/python_Text_extraction/pdf_to_text/scanned_pdf_to_text.py#L53
def preprocess_bad_quality_text(img_path):
    """
    :param img_path: path to image for rescale, convert the color from RGB to Gray, erode, dilate and remove/reduce the noises with a filter
    """
    # https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
    # https://nanonets.com/blog/ocr-with-tesseract/
    # base on: https://towardsdatascience.com/getting-started-with-tesseract-part-ii-f7f9a0899b3f & https://towardsdatascience.com/getting-started-with-tesseract-part-i-2a6a6b1cf75e
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bilateralFilter(img, 5, 75, 75)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    img = cv2.adaptiveThreshold(
        cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    cv2.imwrite(img_path, img)
