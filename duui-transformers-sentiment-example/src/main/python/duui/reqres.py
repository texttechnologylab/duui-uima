from typing import List, Optional
from pydantic import BaseModel

from .sentiment import SentimentSelection
from .uima import UimaSentenceSelection, UimaAnnotationMeta, UimaDocumentModification


# This is the request sent by DUUI to this tool, i.e. the input data
class DUUIRequest(BaseModel):
    """
    """
    selections: List[UimaSentenceSelection]
    lang: str
    doc_len: int
    model_name: str
    batch_size: int
    ignore_max_length_truncation_padding: bool


# This is the response of this tool back to DUUI, i.e. the output data
class DUUIResponse(BaseModel):
    selections: List[SentimentSelection]
    meta: Optional[UimaAnnotationMeta]
    modification_meta: Optional[UimaDocumentModification]
