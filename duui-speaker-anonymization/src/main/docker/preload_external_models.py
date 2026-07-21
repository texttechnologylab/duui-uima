#!/usr/bin/env python3
"""Cache pinned third-party models needed by the anonymization pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


WHISPER_FILES = [
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--whisper-model", required=True)
    parser.add_argument("--whisper-revision", required=True)
    parser.add_argument("--silero-repository", required=True)
    args = parser.parse_args()

    whisper_cache = args.models_dir / "whisper-large-v3"
    torch_cache = args.models_dir / "torch"
    whisper_cache.mkdir(parents=True, exist_ok=True)
    torch_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)

    snapshot_download(
        repo_id=args.whisper_model,
        revision=args.whisper_revision,
        cache_dir=whisper_cache,
        allow_patterns=WHISPER_FILES,
    )
    print(f"Cached {args.whisper_model}@{args.whisper_revision}")

    torch.hub.load(
        repo_or_dir=args.silero_repository,
        model="silero_vad",
        force_reload=False,
        onnx=False,
        skip_validation=True,
        trust_repo=True,
        verbose=False,
    )
    print(f"Cached {args.silero_repository}")


if __name__ == "__main__":
    main()
