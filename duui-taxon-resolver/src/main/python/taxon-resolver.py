import logging
from typing import List, Literal, Self

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from urllib3 import request

import ncbi_api, gbif_api, taxref_loader
from shared_taxon import SharedTaxon, TaxonBase, TaxonProvider

class Settings(BaseSettings):
    annotator_name: str = Field("duui-taxon-resolver", env="ANNOTATOR_NAME")
    annotator_version: str = Field("1.0", env="ANNOTATOR_VERSION")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    execution_mode: Literal["development", "production"] = Field("development", env="EXECUTION_MODE")
    
    class Config:
        env_prefix = "TAXON_RESOLVER_"

settings = Settings()

lua_communication_script_path: str
typesystem_path: str
if settings.execution_mode == "development":
    lua_communication_script_path = "../lua/communication_layer.lua"
    typesystem_path = "../resources/typesystem.xml"
elif settings.execution_mode == "production":
    lua_communication_script_path = "/app/communication_layer.lua"
    typesystem_path = "/app/typesystem.xml"
else:
    raise ValueError(f"Unknown execution mode '{settings.execution_mode}'")

# Init logger
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=settings.log_level,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info("TTLab Taxon Resolver started in %s mode", settings.execution_mode)
logger.info("Name: %s", settings.annotator_name)
logger.info("Version: %s", settings.annotator_version)

logger.info("Loading backbone data for Taxref...")
taxref_loader.initialize_backbone()
logger.info("Taxref backbone data loaded successfully")

def load_communication_script() -> str:
    with open(lua_communication_script_path, "r") as f:
        communication_script = f.read()
        logger.info("Loaded Lua communication script from %s", lua_communication_script_path)
        return communication_script

def load_typesystem() -> str:
    with open(typesystem_path, "r") as f:
        typesystem = f.read()
        logger.info("Loaded type system from %s", typesystem_path)
        return typesystem

lua_communication_script: str | None = load_communication_script() if settings.execution_mode == "production" else None
typesystem: str | None = load_typesystem() if settings.execution_mode == "production" else None

# FastAPI app
app = FastAPI(
    title=settings.annotator_name,
    description="Annotator for resolving and normalizing taxons in documents",
    version=settings.annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    result = lua_communication_script
    if result is None:
        result = load_communication_script()
    return result


# Return typesystem
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    result = typesystem
    if result is None:
        result = load_typesystem()
    return Response(content=result, media_type="application/xml")

class RecognizedTaxonLinking(BaseModel):
    provider: TaxonProvider
    taxon_id: int

    @classmethod
    def from_string(cls, linking_str: str) -> Self:
        try:
            provider_str, taxon_id_str = linking_str.split(":")
            provider: TaxonProvider = provider_str.lower()
            taxon_id = int(taxon_id_str)
            return cls(provider=provider, taxon_id=taxon_id)
        except Exception as e:
            logger.error("Error parsing linking string '%s': %s", linking_str, e)
            raise ValueError(f"Invalid linking string '{linking_str}': {e}")

class RecognizedTaxon(BaseModel):
    begin: int
    end: int
    text: str
    linkings: List[RecognizedTaxonLinking]

class RequestTaxon(BaseModel):
    begin: int
    end: int
    linkings: List[str]

    def to_recognized_taxon(self, document_text: str) -> RecognizedTaxon:
        text = document_text[self.begin:self.end]
        return RecognizedTaxon(
            begin=self.begin,
            end=self.end,
            text=text,
            linkings=[RecognizedTaxonLinking.from_string(linking_str) for linking_str in self.linkings]
        )

class DuuiRequest(BaseModel):
    taxa: List[RequestTaxon]
    document_text: str

    @property
    def recognized_taxa(self) -> List[RecognizedTaxon]:
        return [taxon.to_recognized_taxon(self.document_text) for taxon in self.taxa]
    
class ExportedTaxon(BaseModel):
    begin: int
    end: int
    text: str
    resolved_linkings: List[SharedTaxon]

class DuuiResponse(BaseModel):
    taxa: List[ExportedTaxon]

def resolve_taxon_linking(linking: RecognizedTaxonLinking) -> TaxonBase:

    try:
        match linking.provider:
            case "ncbi":
                return ncbi_api.NcbiTaxon.from_tax_id(linking.taxon_id)
            case "gbif":
                return gbif_api.get_taxon(linking.taxon_id)
            case "taxref":
                return taxref_loader.taxon_from_id(linking.taxon_id)
            case _:
                raise ValueError(f"Unknown taxon provider '{linking.provider}'")
    except Exception as e:
        logger.error(e)


def resolve_taxon_linkings(linkings: List[RecognizedTaxonLinking]) -> List[TaxonBase]:
    return [resolve_taxon_linking(linking) for linking in linkings]

def resolve_recognized_taxon(recognized_taxon: RecognizedTaxon) -> ExportedTaxon:
    resolved_linkings = resolve_taxon_linkings(recognized_taxon.linkings)
    return ExportedTaxon(
        begin=recognized_taxon.begin,
        end=recognized_taxon.end,
        text=recognized_taxon.text,
        resolved_linkings=[linking.as_shared() for linking in resolved_linkings if linking is not None]
    )

def resolve_recognized_taxa(recognized_taxa: List[RecognizedTaxon]) -> List[ExportedTaxon]:
    return [resolve_recognized_taxon(recognized_taxon) for recognized_taxon in recognized_taxa]

@app.post("/v1/process")
async def post_process(request: DuuiRequest) -> DuuiResponse:
    recognized_taxa = request.recognized_taxa
    logger.debug(recognized_taxa)
    resolved_taxa = resolve_recognized_taxa(recognized_taxa)
    logger.debug("Resolved %d taxons", len(resolved_taxa))
    return DuuiResponse(taxa=resolved_taxa)