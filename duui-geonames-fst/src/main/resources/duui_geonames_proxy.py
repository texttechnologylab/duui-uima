#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any, Dict

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse


BACKEND_URL = os.environ.get("GEONAMES_BACKEND", "http://127.0.0.1:9715")
RESOURCE_DIR = Path(os.environ.get("DUUI_RESOURCE_DIR", Path(__file__).resolve().parent))
COMMUNICATION_LAYER = Path(
    os.environ.get("DUUI_COMMUNICATION_LAYER", RESOURCE_DIR / "communication_layer.lua")
)

app = FastAPI(title="DUUI GeoNames FST Proxy")


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "duui-geonames-fst-proxy",
        "backend": BACKEND_URL,
        "communication_layer": str(COMMUNICATION_LAYER),
        "description": (
            "Proxy for DUUI serialize/deserialize mode. "
            "Serves the Lua communication layer, adds Content-Type: application/json "
            "and restores begin/end offsets."
        ),
    }


@app.get("/v1/communication_layer")
async def communication_layer() -> Response:
    """DUUI fetches the Lua communication layer via GET before processing."""
    try:
        content = COMMUNICATION_LAYER.read_text(encoding="utf-8")
    except FileNotFoundError:
        return PlainTextResponse(
            f"communication_layer.lua not found at {COMMUNICATION_LAYER}",
            status_code=500,
        )

    return PlainTextResponse(content, status_code=200, media_type="text/plain")


@app.get("/{path:path}")
async def forward_get(path: str, request: Request) -> Response:
    """Forward non-Lua GET requests to the real backend."""
    return await forward_raw_get(request, f"/{path}")


@app.post("/")
async def process_root(request: Request) -> Response:
    return await forward_json_post(request, "/")


@app.post("/{path:path}")
async def process_path(path: str, request: Request) -> Response:
    return await forward_json_post(request, f"/{path}")


async def forward_raw_get(request: Request, path: str) -> Response:
    backend_url = BACKEND_URL.rstrip("/") + path

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            backend_response = await client.get(
                backend_url,
                params=dict(request.query_params),
                headers={"Accept": request.headers.get("accept", "*/*")},
            )
    except Exception as exc:
        return PlainTextResponse(
            f"Proxy error while forwarding GET to {backend_url}: {exc}",
            status_code=502,
        )

    return Response(
        content=backend_response.content,
        status_code=backend_response.status_code,
        media_type=backend_response.headers.get("content-type", "text/plain"),
    )


async def forward_json_post(request: Request, path: str) -> Response:
    body = await request.body()

    if not body:
        return PlainTextResponse("Empty request body", status_code=400)

    try:
        request_json = await request.json()
    except Exception as exc:
        return PlainTextResponse(f"Invalid JSON request body: {exc}", status_code=400)

    reference_offsets = build_reference_offset_map(request_json)
    backend_url = BACKEND_URL.rstrip("/") + path

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            backend_response = await client.post(
                backend_url,
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
    except Exception as exc:
        return PlainTextResponse(
            f"Proxy error while forwarding POST to {backend_url}: {exc}",
            status_code=502,
        )

    if backend_response.status_code < 200 or backend_response.status_code >= 300:
        return Response(
            content=backend_response.content,
            status_code=backend_response.status_code,
            media_type=backend_response.headers.get("content-type", "text/plain"),
        )

    try:
        response_json = backend_response.json()
    except Exception:
        return Response(
            content=backend_response.content,
            status_code=backend_response.status_code,
            media_type=backend_response.headers.get("content-type", "application/json"),
        )

    response_json = enrich_response_with_offsets(response_json, reference_offsets)

    return JSONResponse(
        content=response_json,
        status_code=backend_response.status_code,
    )


def build_reference_offset_map(request_json: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}

    for query in request_json.get("queries", []):
        reference = query.get("reference")
        begin = query.get("begin")
        end = query.get("end")

        if reference is None or begin is None or end is None:
            continue

        result[str(reference)] = {
            "begin": int(begin),
            "end": int(end),
        }

    return result


def enrich_response_with_offsets(
        response_json: Dict[str, Any],
        reference_offsets: Dict[str, Dict[str, int]],
) -> Dict[str, Any]:
    for result in response_json.get("results", []):
        reference = result.get("reference")

        if reference is None:
            continue

        offsets = reference_offsets.get(str(reference))

        if offsets is None:
            continue

        result["begin"] = offsets["begin"]
        result["end"] = offsets["end"]

    return response_json


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9714"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )