from typing import List, Optional
from pydantic import BaseModel

from .sentiment import SentimentSelection
from .uima import UimaSentenceSelection, UimaAnnotationMeta, UimaDocumentModification


class DUUIRequest(BaseModel):
    selections: List[UimaSentenceSelection]
    lang: str
    doc_len: int


class DUUIResponse(BaseModel):
    selections: List[SentimentSelection]
    meta: Optional[UimaAnnotationMeta]
    modification_meta: Optional[UimaDocumentModification]
