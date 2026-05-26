import asyncio
import logging
import os
import shutil
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Final

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.logger import logger
from starlette.background import BackgroundTask

LOGGING_CONFIG: Final[dict] = uvicorn.config.LOGGING_CONFIG
LOGGING_CONFIG["loggers"][""] = {
    "handlers": ["default"],
    "level": "INFO",
    "propagate": False,
}
logging.config.dictConfig(LOGGING_CONFIG)


GNFINDER_PATH: Final[Path] = Path(
    os.environ.get("GNFINDER_PATH", shutil.which("gnfinder"))
)
STARTUP_DELAY: Final[int] = int(os.environ.get("STARTUP_DELAY", "2"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "gnfinder_process"):
        logger.info("[startup] Starting GNFinder process")
        app.state.gnfinder_process = await asyncio.create_subprocess_exec(
            GNFINDER_PATH, "-p", "8999"
        )

    try:
        # Wait for the gnfinder server to start
        # If the server exits during this time, raise a RuntimeError
        async with asyncio.timeout(STARTUP_DELAY):
            if exit_code := await app.state.gnfinder_process.wait():
                raise RuntimeError(
                    f"GNFinder server exited unexpectedly with code {exit_code}"
                )
    except TimeoutError:
        pass

    async with httpx.AsyncClient(base_url="http://localhost:8999/") as gnfinder_client:
        try:
            # Check if the GNFinder server is running
            (await gnfinder_client.get("/api/v1/ping")).raise_for_status()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=httpx.codes.SERVICE_UNAVAILABLE,
                detail="Could not connect to GNFinder server",
            ) from e

        yield {
            "gnfinder_client": gnfinder_client,
        }

    try:
        if hasattr(app.state, "gnfinder_process"):
            logger.info("[shutdown] Terminating GNFinder process")
            app.state.gnfinder_process.terminate()

            # Wait for the process to terminate
            async with asyncio.timeout(5):
                await app.state.gnfinder_process.communicate()
    except TimeoutError as e:
        raise RuntimeError(
            "GNFinder process did not terminate in a timely manner"
        ) from e
    except ProcessLookupError:
        # expected during shutdown, process already terminated by context manager, but required for reloading
        pass
    finally:
        del app.state.gnfinder_process


app = FastAPI(lifespan=lifespan)


#####


with open("type_system.xml", "r") as f:
    TYPE_SYSTEM_XML: Final[str] = f.read()


@app.get("/v1/typesystem")
def get_type_system():
    return Response(TYPE_SYSTEM_XML, media_type="application/xml")


with open("communication_layer.lua", "r") as f:
    LUA_COMMUNICATION_LAYER: Final[str] = f.read()


@app.get("/v1/communication_layer")
def get_communication_layer():
    return Response(LUA_COMMUNICATION_LAYER, media_type="text/x-lua")


#####


@app.get("/api/v1/{path:path}")
@app.post("/api/v1/{path:path}")
async def forward_api_request(request: Request):
    client: httpx.AsyncClient = request.state.gnfinder_client
    try:
        response: httpx.Response = await client.send(
            client.build_request(
                request.method,
                url=httpx.URL(
                    path=request.url.path, query=request.url.query.encode("utf-8")
                ),
                headers=[(k, v) for k, v in request.headers.raw if k != b"host"],
                content=request.stream(),
            )
        )
        response_headers = {k.lower(): v for k, v in response.headers.items()}
        headers = (
            {"content-type": conent_type}
            if (conent_type := response_headers.get("content-type")) is not None
            else {}
        )
        return Response(
            headers=headers,
            content=response.content,
            status_code=response.status_code,
            background=BackgroundTask(response.aclose),
        )
    except httpx.HTTPStatusError as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=e.response.status_code,
            detail=traceback.format_exc(),
        ) from e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=traceback.format_exc(),
        ) from e
