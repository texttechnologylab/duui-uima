from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from .uima import UimaAnnotationMeta, UimaDocumentModification


class TextImagerRequest(BaseModel):
    docs: List[str]
    model_name: str
    lang: str
    summary_length: int
    parameters: Optional[dict]


class TextImagerResponse(BaseModel):
    summaries: List[str]
    meta: Optional[UimaAnnotationMeta]
    modification_meta: Optional[UimaDocumentModification]
