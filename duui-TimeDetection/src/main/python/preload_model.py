from __future__ import annotations

import os

from time_recognition_backend import apply_model_settings, resolve_model_name


def main() -> None:
    model_name = os.environ.get("MODEL_NAME", "microsoft")
    model_specname = os.environ.get("MODEL_SPECNAME", "")
    model_source = os.environ.get("MODEL_SOURCE", "")
    model_lang = os.environ.get("MODEL_LANG", "")
    model_version = os.environ.get("MODEL_VERSION", "latest")

    alias, cfg = resolve_model_name(model_name)
    cfg = apply_model_settings(
        cfg,
        model_specname=model_specname,
        model_source=model_source,
        model_lang=model_lang,
    )

    backend = cfg.get("backend")
    if backend != "hf_token_classification":
        print(f"No build-time model preload needed for backend {backend}.")
        return

    model_id = cfg.get("model_id")
    if not model_id:
        raise RuntimeError(
            "MODEL_SPECNAME or a registry model_id is required for Hugging Face backends."
        )

    revision = None if model_version in {"", "latest"} else model_version

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    AutoTokenizer.from_pretrained(model_id, use_fast=True, revision=revision)
    AutoModelForTokenClassification.from_pretrained(model_id, revision=revision)

    print(f"Preloaded Hugging Face model {model_id} revision={revision or 'default'} for alias {alias}.")


if __name__ == "__main__":
    main()
