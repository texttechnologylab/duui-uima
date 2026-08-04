#!/usr/bin/env python3
"""Download and verify the fixed IMS-Toucan model artifacts."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://github.com/DigitalPhonetics/IMS-Toucan/releases/download/v2.5"
MODEL_SHA256 = {
    "embedding_function.pt": "731ba38ccc5fed2c1d9b4d5d0b1ba7856b2fe68168fb19c77d1c21d1f94ce542",
    "embedding_gan.pt": "20f329d19ab77e58ec9527527636178b3451c5af27ecdfa30a22bd01ec76b9e5",
    "aligner.pt": "b3f653d344d395a6f45cd3b7540aa34f21f32fd4cfc7de0af920cb8ae8dc536e",
    "ToucanTTS_Meta.pt": "595e0199e97cc41d43c93515d3a04c0b32b10d6eb04671ded31370c72a69501b",
    "Avocodo.pt": "9aec4a4f3b912443abc7436a4e41ada3c07f16f3aa515b5817e79ede471702a2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required model is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}"
        )


def download(url: str, destination: Path, expected: str, attempts: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    if destination.exists():
        verify(destination, expected)
        print(f"Verified existing {destination.name}")
        return

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            digest = hashlib.sha256()
            request = urllib.request.Request(
                url, headers={"User-Agent": "duui-speaker-anonymization-model-fetcher/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Checksum mismatch for {destination.name}: expected {expected}, got {actual}"
                )
            os.replace(temporary, destination)
            print(f"Downloaded and verified {destination.name}")
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(attempt * 2)

    raise RuntimeError(f"Unable to download {destination.name} after {attempts} attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    for filename, expected in MODEL_SHA256.items():
        path = args.destination / filename
        if args.verify_only:
            verify(path, expected)
            print(f"Verified {filename}")
        else:
            download(f"{args.base_url.rstrip('/')}/{filename}", path, expected)


if __name__ == "__main__":
    main()

