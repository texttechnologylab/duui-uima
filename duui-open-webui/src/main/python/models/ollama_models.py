from pydantic import BaseModel
from typing import Optional, List

class OllamaConfig(BaseModel):
    host: str = "http://localhost"
    port: int = 11434
    auth_token: Optional[str] = None


class OllamaRequest(BaseModel):
    model: str
    prompt: str
    system_prompt: Optional[str] = None
    images: Optional[List[str]] = None  # Base64-encoded
    audio: Optional[str] = None         # Base64-encoded
    video: Optional[str] = None         # Base64-encoded

class OllamaResponse(BaseModel):
    response: str
    model: str
    status: str
    error: Optional[str] = None
