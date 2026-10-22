import logging
import duui_logging
from duui_logging import log_info, log_warn, log_error, log_debug

from typing import List, Optional
from pydantic import BaseModel

from .sentiment import SentimentSelection
from .uima import UimaSentenceSelection, UimaAnnotationMeta, UimaDocumentModification


class TextImagerRequest(BaseModel):
    selections: List[UimaSentenceSelection]
    lang: str
    doc_len: int


class TextImagerResponse(BaseModel):
    selections: List[SentimentSelection]
    meta: Optional[UimaAnnotationMeta] = None
    modification_meta: Optional[UimaDocumentModification] = None