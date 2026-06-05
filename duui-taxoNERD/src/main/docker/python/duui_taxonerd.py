from typing import List, Optional
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from pydantic import BaseModel
from taxonerd import TaxoNERD
import uvicorn
import time
import threading
import os
from cassis import *


ner: TaxoNERD = None
initialized: bool = False
init_lock = threading.Lock()



class ModelConfig(BaseModel):
    model: str = os.getenv("TAXONERD_MODEL", "en_ner_eco_md")
    linker: str = os.getenv("TAXONERD_LINKER", "gbif_backbone")
    threshold: float = float(os.getenv("TAXONERD_THRESHOLD", "0.7"))
    exclude: list = [
        "tagger",
        "parser"
#         "taxo_abbrev_detector",
#         "taxon_linker",
#         "pysbd_sentencizer"
    ]

config = ModelConfig()

perf_stats = {
    "init_time": None,
    "warmup_time": None,
    "requests": 0,
    "total_inference_time": 0.0
}

class GBIF(BaseModel):
    id: str
    value: str
    propability: float


# Taxon
class Taxon(BaseModel):
    begin: int
    end: int
    comment: Optional[List[GBIF]]


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The text to process
    text: str
    # Linking parameter
    linking: str
    # Threshold
    threshold: float
    #
    exclude: List[str]
    #
    model: str


# Response of this annotator
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # List of Taxon
    taxons: List[Taxon]


def analyse(text, ner):

    start = time.perf_counter()

    result = ner.find_in_text(text)
    # print(result)

    if result.empty:
        return []

    offsets_col = result["offsets"].values
    entities_col = result["entity"].values

    taxons = []

    append_taxon = taxons.append  # local binding (micro-optimization)

    for offsets, entities in zip(offsets_col, entities_col):

        # schneller als split + indexing mehrfach
        parts = offsets.split(" ")
        begin = int(parts[1])
        end = int(parts[2])

        # entries effizient bauen
        entries = [
            {
                "id": str(e[0]),
                "value": str(e[1]),
                "propability": float(e[2])
            }
            for e in entities
        ]

        append_taxon({
            "begin": begin,
            "end": end,
            "comment": entries
        })

    end = time.perf_counter()

    # update runtime stats
    perf_stats["requests"] += 1
    perf_stats["total_inference_time"] += (end - start)

#     print(f"⚡ Inference time: {end - start:.4f}s")

    return taxons



def get_perf_summary():

    if perf_stats["requests"] == 0:
        return {"status": "no requests yet"}

    avg = perf_stats["total_inference_time"] / perf_stats["requests"]

    return {
        "init_time": perf_stats["init_time"],
        "warmup_time": perf_stats["warmup_time"],
        "requests": perf_stats["requests"],
        "avg_inference_time": avg
    }


@asynccontextmanager
async def lifespan(app):

    global ner, initialized

    print("🔥 Starting TaxoNERD initialization...")

    t0 = time.time()

    with init_lock:
        ner = TaxoNERD(prefer_gpu=True)

        ner.load(
            model=config.model,
            linker=config.linker,
            threshold=config.threshold,
            exclude=config.exclude
        )

        # GPU warmup (wichtig für first request latency)
        ner.find_in_text("Homo sapiens")

        initialized = True

    print(f"✅ TaxoNERD ready in {time.time() - t0:.2f}s")

    yield

    print("🧹 Shutdown complete")

def get_ner():
    return ner

app = FastAPI(
    lifespan=lifespan,
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title="TaxoNERD",
    description="TaxoNERD implementation for TTLab DUUI",
    version="0.1",
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "Giuseppe Abrami, TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "abrami@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

# Load the Lua communication script
communication = "communication.lua"
with open(communication, 'rb') as f:
    communication = f.read().decode("utf-8")

# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'types.xml'
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)


# Get input / output of the annotator
@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:

    json_item = {
        "inputs": [],
        "outputs": ["org.texttechnologylab.annotation.type.Taxon"]
    }

    json_compatible_item_data = jsonable_encoder(json_item)
    return JSONResponse(content=json_compatible_item_data)

# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")
    return Response(
        content=xml_content,
        media_type="application/xml"
    )


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return communication

# Process request from DUUI
@app.post("/v1/process")
async def post_process(request: DUUIRequest) -> DUUIResponse:

    model = get_ner()

    taxons = await run_in_threadpool(analyse, request.text, model)

#     print(get_perf_summary())

    # Return data as JSON
    return DUUIResponse(taxons=taxons)



if __name__ == "__main__":

    uvicorn.run("duui_taxonerd:app", host="0.0.0.0", port=9714, workers=1)
