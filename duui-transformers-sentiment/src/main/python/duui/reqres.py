from typing import List, Optional
from pydantic import BaseModel

from .sentiment import SentimentSelection
from .uima import UimaSentenceSelection, UimaAnnotationMeta, UimaDocumentModification


class TextImagerRequest(BaseModel):
    selections: List[UimaSentenceSelection]
    lang: str
    doc_len: int
    model_name: str


class TextImagerResponse(BaseModel):
    selections: List[SentimentSelection]
    meta: Optional[UimaAnnotationMeta]
    modification_meta: Optional[UimaDocumentModification]
