"""
duui_ocr_server.py

FastAPI server that wraps vision-language models and exposes them as a
DUUI-compatible annotator component. You send it images (base64 or file
paths), it sends back OCR text.

ITERATION HISTORY:

  v1:  PaddleOCR-VL-1.5 only. One model, everything in one huge function,
       worked but was impossible to extend. I knew from the start we'd
       need to support more models, so I had to improve on this version
       even though it technically ran fine. The problem was architectural,
       not functional.

  v2:  Tried to add microsoft/trocr-base-printed as a second model.
       Spent two days on this before realizing TrOCR is a fundamentally
       different kind of model. It uses VisionEncoderDecoderModel instead
       of AutoModelForImageTextToText, needs its own TrOCRProcessor
       instead of AutoProcessor, has no concept of chat templates or
       text prompts, and this is the real killer: it only works on
       single text-line images :(. You literally have to pre-crop every
       line of text before feeding it in. It doesn't do full-page OCR.
       My whole infrastructure assumes you hand the model a page and get
       text back. TrOCR assumes someone else already found the text
       lines for you. I couldn't reconcile these two approaches without
       rewriting everything into two completely separate pipelines, and
       at that point what's the shared infrastructure even for?
       The abandoned TrOCR backend code is still in this file, commented
       out, as proof of "concept".

  v3:  Added zai-org/GLM-OCR instead. This worked almost immediately
       because GLM-OCR is architecturally the same *kind* of model as
       PaddleOCR-VL: it's a vision-language model that uses
       AutoModelForImageTextToText, supports AutoProcessor with chat
       templates, accepts text prompts alongside images, and does
       full-page OCR. The backend pattern I'd already built for PaddleOCR
       fit GLM-OCR with only minor adjustments. Sometimes the answer
       isn't "write more code," it's "pick a compatible model."

Heavy lifting on the model loading, batching, and the generate() call
was done with GitHub Copilot. I understand the flow but some of the
torch-specific idioms (inference_mode, bfloat16, cache eviction) are
things I looked up rather than knew from experience.

The DUUI integration layer (typesystem, lua script, endpoints) is
mostly lifted from existing DUUI annotator examples in the TTLab repo:
https://github.com/texttechnologylab/DockerUnifiedUIMAInterface

Last meaningful edit: Feb 2026
"""

from __future__ import annotations

import base64
import gc
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Dict, List, Optional, Union

import torch
from cassis import load_typesystem
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from PIL import Image as PILImage
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse, PlainTextResponse, Response
from transformers import AutoModelForImageTextToText, AutoProcessor

# -- v2: TrOCR imports --
# ABANDONED.
# TrOCR needs its own model class and processor class. It can't use
# the Auto* classes that PaddleOCR-VL and GLM-OCR share.
#
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -- Registry --
# SOLID. This is just a dictionary.
#
# Each model we support gets an entry here with its metadata.
# "task_prompts" maps a task name to the literal string the model
# expects as its instruction. I got these prompt strings from the
# respective model cards on HuggingFace:
#   - https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5
#   - https://huggingface.co/zai-org/GLM-OCR
#
# If you add a new model, you add it here and write a backend class
# for it below. "backend" is just a string key that maps to a class
# in BACKEND_MAP at the bottom of the backends section.

MODEL_REGISTRY = {
    "PaddlePaddle/PaddleOCR-VL-1.5": {
        "source": "https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5",
        "lang": "multi",
        "version": "2026-01-28",
        "tasks": ["ocr", "table", "formula", "chart", "spotting", "seal"],
        "task_prompts": {
            "ocr": "OCR:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
            "chart": "Chart Recognition:",
            "spotting": "Spotting:",
            "seal": "Seal Recognition:",
        },
        "backend": "paddleocr",
    },
    # -- v2: TrOCR registry entry --
    # ABANDONED. I had this in the registry for about 6 hours before
    # I realized it was never going to work with the shared backend.
    #
    # "microsoft/trocr-base-printed": {
    #     "source": "https://huggingface.co/microsoft/trocr-base-printed",
    #     "lang": "en",       # TrOCR is English-only, unlike the others
    #     "version": "2021-09-21",
    #     "tasks": ["ocr"],   # only OCR, no table/formula/chart support
    #     "task_prompts": {
    #         # TrOCR doesn't actually use text prompts at all.
    #         # It just takes pixel_values and generates text directly.
    #         # I put this here to fit the registry schema but it's
    #         # meaningless, the TrOCR backend ignores it.
    #         "ocr": "",
    #     },
    #     "backend": "trocr",
    # },
    "zai-org/GLM-OCR": {
        "source": "https://huggingface.co/zai-org/GLM-OCR",
        "lang": "multi",
        "version": "2026-02-09",
        "tasks": ["ocr", "table", "formula"],
        "task_prompts": {
            "ocr": "Text Recognition:",
            "table": "Table Recognition:",
            "formula": "Formula Recognition:",
        },
        "backend": "glmocr",
    },
}

# Just collects every unique task string across all models.
# The sorted() is cosmetic, I like alphabetical order in API docs.
ALL_SUPPORTED_TASKS = sorted(
    {t for m in MODEL_REGISTRY.values() for t in m["tasks"]}
)

# -- Settings & globals --
# BORROWED. pydantic-settings pattern from TTLab's other DUUI components.
# Source: https://github.com/texttechnologylab/DockerUnifiedUIMAInterface
#
# The idea is that all config comes from environment variables so the
# Docker container can be parameterized at runtime. BaseSettings does
# the env-var-to-field mapping automatically, which I didn't know before.


class Settings(BaseSettings):
    duui_ocr_annotator_name: str
    duui_ocr_annotator_version: str
    duui_ocr_log_level: str
    duui_ocr_model_cache_size: int = 1  # how many models to keep loaded


settings = Settings()
logging.basicConfig(level=settings.duui_ocr_log_level)
logger = logging.getLogger(__name__)

# COPILOT. I asked Copilot "how to pick GPU vs CPU and set dtype for
# transformers inference" and this is essentially what it gave me.
# bfloat16 is a half-precision float that saves VRAM. I *think* it's
# fine for inference but not for training? Either way it works here.
# On CPU we fall back to float32 because bfloat16 support on CPU is
# patchy depending on the hardware.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32
logger.info("Using device: %s", DEVICE)

# Only one thread can use the model at a time. Without this lock,
# concurrent requests can corrupt the GPU state and you get cryptic
# CUDA errors. Learned that the hard way during testing.
model_lock = Lock()

# BORROWED, DUUI boilerplate. Every DUUI annotator needs a UIMA type
# system (XML) and a Lua communication script. These are loaded once
# at startup. The format is dictated by the DUUI framework.
with open("TypeSystemOCR.xml", "rb") as f:
    typesystem = load_typesystem(f)
with open("duui_ocr.lua", "rb") as f:
    lua_communication_script = f.read().decode("utf-8")

# -- Schemas --
# SOLID. These are just data shapes for the API. Pydantic validates
# incoming JSON against these classes automatically, which is genuinely
# one of the nicest things about FastAPI.
#
# "begin" and "end" are character offsets in the original UIMA document.
# They travel with the image so we can attach the OCR result back to
# the right spot in the document.


class ImageInput(BaseModel):
    src: str  # base64-encoded image data or a file path
    begin: int
    end: int


class OCRResult(BaseModel):
    text: str  # the recognized text
    task: str  # which task produced this ("ocr", "table", etc.)
    begin: int
    end: int


class OCRRequest(BaseModel):
    images: List[ImageInput]
    lang: str
    doc_len: int
    model_name: str
    task: str = "ocr"
    max_new_tokens: int = 1024  # upper bound on model output length


class OCRResponse(BaseModel):
    ocr_results: List[OCRResult]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str
    errors: List[str]  # we collect errors instead of crashing
    config: Dict[str, Union[str, int, bool]]


# BORROWED, DUUI documentation schema. Every annotator must describe
# itself through this endpoint. Copied from existing annotators.
class TextImagerDocumentation(BaseModel):
    annotator_name: str
    version: str
    implementation_lang: Optional[str] = None
    meta: Optional[dict] = None
    parameters: Optional[dict] = None


# -- Helpers --


def decode_image(src: str) -> PILImage.Image:
    """
    SOLID. Takes either a file path or a base64 string and gives
    back a PIL image. The .convert("RGB") is important because some
    PNGs come in as RGBA or palette mode and the models choke on that.
    I found that out after a very confusing afternoon of "why does
    this work on JPEGs but not PNGs?"
    """
    if os.path.isfile(src):
        return PILImage.open(src).convert("RGB")
    return PILImage.open(BytesIO(base64.b64decode(src))).convert("RGB")


def to_device(mapping: dict) -> dict:
    """
    COPILOT. Moves all tensors in a dict to the target device (GPU/CPU).
    Copilot generated this as a one-liner dict comprehension. I expanded
    it for readability. The isinstance check is there because the
    processor output dict also contains non-tensor values (like
    attention masks as lists sometimes?) and you can't call .to() on those.
    """
    return {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in mapping.items()
    }


def generate(model, inputs: dict, max_new_tokens: int):
    """
    COPILOT. wraps model.generate() with the settings we want.
    Prompt was roughly "generate from a transformers model with no
    sampling deterministic output."

    - inference_mode: faster than no_grad, Copilot's suggestion.
      I *think* it disables autograd more aggressively.
    - do_sample=False: deterministic output, same image = same text.
    - use_cache=True: something about reusing intermediate computations
      during token generation. Makes it faster. I don't fully understand
      the KV-cache mechanism but every example I've seen sets this to True.
    """
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )


# -- Backends --
# This is the part I'm least confident about architecturally.
# The idea: each model family has slightly different ways of building
# the input prompt and decoding the output. So each one gets its own
# "backend" class that knows how to talk to that specific model.
#
# The abstract base class defines the interface. Subclasses fill in
# the details. I learned this pattern from the original code in the
# DUUI repo.
#
# v1 had just PaddleOCR, no abstraction needed.
# v2 is where I introduced the ABC because I thought TrOCR would be
# a second subclass. It wasn't :((. TrOCR's interface was too different.
# v3 kept the ABC because GLM-OCR actually fits it perfectly.
# So the abstraction turned out to be useful, just not for the model
# I originally designed it for.


class OCRBackend(ABC):
    def __init__(self, model_name: str, model, processor):
        self.model_name = model_name
        self.model = model
        self.processor = processor
        self.meta = MODEL_REGISTRY[model_name]

    def get_prompt(self, task: str) -> str:
        """
        SOLID. Looks up the prompt string for a given task.
        Falls back to the "ocr" prompt if the task isn't found,
        which is a safety net that probably shouldn't be needed
        since we validate tasks earlier. But just in case.
        """
        prompts = self.meta["task_prompts"]
        return prompts.get(task, prompts.get("ocr", "OCR:"))

    @abstractmethod
    def run_single(
        self, image: PILImage.Image, task: str, max_new_tokens: int
    ) -> str: ...

    def run_batch(
        self,
        images: List[PILImage.Image],
        task: str,
        max_new_tokens: int,
    ) -> List[str]:
        """
        WORKS. Tries batch processing first, and if that fails for
        any reason (OOM, padding issues, whatever) falls back to
        processing images one at a time. This saved me during testing
        when batch processing would randomly fail on certain image
        size combinations. The sequential fallback is slower but
        at least it doesn't crash the whole request.
        """
        try:
            return self._run_batch_impl(images, task, max_new_tokens)
        except Exception as e:
            logger.warning("Batch failed, falling back to sequential: %s", e)
            return [
                self.run_single(img, task, max_new_tokens) for img in images
            ]

    def _run_batch_impl(
        self,
        images: List[PILImage.Image],
        task: str,
        max_new_tokens: int,
    ) -> List[str]:
        """Default: just loops. Subclasses override with real batching."""
        return [
            self.run_single(img, task, max_new_tokens) for img in images
        ]


class PaddleOCRBackend(OCRBackend):
    """
    BORROWED + COPILOT. Backend for PaddlePaddle/PaddleOCR-VL-1.5.
    This was the first model I got working (v1). The chat template
    pattern (apply_chat_template) comes from the HuggingFace model
    card example:
    https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

    Copilot helped me adapt it for batch processing. The _decode
    method with the .split(chat_text)[-1] trick is from the model
    card too. Yhe model repeats the prompt in its output so you
    have to strip it. Took me a while to figure out why I was
    getting the prompt text echoed back in my results.
    """

    def _chat_text(self, task: str) -> str:
        # Builds the chat-formatted prompt string the model expects.
        # The structure with "role" / "content" / list of dicts is
        # the HuggingFace chat template convention.
        # {"type": "image"} is a placeholder — the actual pixel data
        # gets passed separately to the processor.
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.get_prompt(task)},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _decode(self, generated, chat_text: str) -> List[str]:
        # FRAGILE. The split-on-prompt-text approach assumes the
        # model always echoes the prompt. If a future model version
        # changes this behavior, results will break silently.
        decoded = self.processor.batch_decode(
            generated, skip_special_tokens=True
        )
        return [r.split(chat_text)[-1].strip() for r in decoded]

    def run_single(self, image, task, max_new_tokens):
        text = self._chat_text(task)
        inputs = to_device(
            self.processor(text=[text], images=[image], return_tensors="pt")
        )
        out = generate(self.model, inputs, max_new_tokens)
        return self._decode(out, text)[0]

    def _run_batch_impl(self, images, task, max_new_tokens):
        # Same as run_single but we pass all images at once with
        # padding=True so the processor pads shorter sequences to
        # match the longest one. Faster on GPU because it processes
        # in parallel (I think).
        text = self._chat_text(task)
        inputs = to_device(
            self.processor(
                text=[text] * len(images),
                images=images,
                return_tensors="pt",
                padding=True,
            )
        )
        out = generate(self.model, inputs, max_new_tokens)
        return self._decode(out, text)



# v2 ABANDONED: TrOCR Backend
#
# I spent a full weekend trying to make this work. Leaving it here
# commented out as documentation of why it failed, in case anyone
# else gets the same idea.
#
# The core problem: TrOCR (microsoft/trocr-base-printed) is a
# VisionEncoderDecoderModel, not an AutoModelForImageTextToText.
#
# Our whole pipeline sends full page images. TrOCR expects someone
# to have already detected and cropped individual text lines. I'd
# need to add a whole text detection step before TrOCR, basically
# building a separate pipeline.
#
# I also couldn't get it to load through AutoModelForImageTextToText
# without it throwing architecture mismatch errors. Copilot kept
# suggesting workarounds that compiled but produced garbage output.
#
# Source that finally made me understand the difference:
# https://huggingface.co/docs/transformers/en/model_doc/trocr
# https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/vision-encoder-decoder
# Also this HF discussion where someone asks the same question I had:
# https://huggingface.co/microsoft/trocr-base-printed/discussions/3
#
# class TrOCRBackend(OCRBackend):
#     """
#     ABANDONED, backend for microsoft/trocr-base-printed.
#
#     This doesn't actually inherit from OCRBackend cleanly because
#     the interface is too different. I tried to force it to fit by
#     ignoring the task parameter and skipping the prompt, but the
#     real issue is deeper: TrOCR only does single-line OCR.
#
#     Model card: https://huggingface.co/microsoft/trocr-base-printed
#     Paper: https://arxiv.org/abs/2109.10282
#     """
#
#     def __init__(self, model_name: str, model, processor):
#         # Can't call super().__init__() cleanly because the parent
#         # expects self.processor to have apply_chat_template(), which
#         # TrOCRProcessor doesn't have. Already a bad sign.
#         self.model_name = model_name
#         self.model = model
#         self.processor = processor
#         self.meta = MODEL_REGISTRY[model_name]
#
#     def run_single(self, image, task, max_new_tokens):
#         # TrOCR ignores the task parameter entirely. It only does OCR.
#         # No table recognition, no formula recognition, nothing.
#         #
#         # The processor here is TrOCRProcessor, which only takes images.
#         # No text= argument. No chat template. Just pixel_values.
#         pixel_values = self.processor(
#             images=image, return_tensors="pt"
#         ).pixel_values.to(DEVICE)
#
#         with torch.inference_mode():
#             generated_ids = self.model.generate(
#                 pixel_values,
#                 max_new_tokens=max_new_tokens,
#             )
#
#         return self.processor.batch_decode(
#             generated_ids, skip_special_tokens=True
#         )[0]
#
#     def _run_batch_impl(self, images, task, max_new_tokens):
#         # FRAGILE: TrOCR batching. I got this working but the results
#         # were garbage on full-page images. The model would output
#         # random fragments or repeat the same word over and over.
#         #
#         # In hindsight this is obvious: the model was trained on
#         # cropped single-line images at 384x384 resolution.
#         pixel_values = self.processor(
#             images=images, return_tensors="pt", padding=True
#         ).pixel_values.to(DEVICE)
#
#         with torch.inference_mode():
#             generated_ids = self.model.generate(
#                 pixel_values,
#                 max_new_tokens=max_new_tokens,
#             )
#
#         return self.processor.batch_decode(
#             generated_ids, skip_special_tokens=True
#         )
#
# End of abandoned TrOCR code.


class GlmOCRBackend(OCRBackend):
    """
    BORROWED + COPILOT. Backend for zai-org/GLM-OCR (v3 addition).

    After the TrOCR failure I was nervous about adding another model,
    but GLM-OCR turned out to be almost suspiciously easy to integrate.
    The reason: it's the same *kind* of model as PaddleOCR-VL.

    Both are vision-language models built on the
    AutoModelForImageTextToText architecture. Both use AutoProcessor
    with chat templates. Both accept full-page images with text prompts.
    The only real differences are in how the messages dict is structured
    and how you decode the output.

    Specifically, why GLM-OCR works where TrOCR didn't:
      1. GLM-OCR loads with AutoModelForImageTextToText: same class
         as PaddleOCR-VL. No special imports needed.
      2. GLM-OCR's processor supports apply_chat_template() so the
         prompt-building pattern from OCRBackend.get_prompt() just works.
      3. GLM-OCR handles full document pages natively. It was designed
         for "complex document understanding" (their words). No need
         to pre-crop text lines.
      4. GLM-OCR supports multiple tasks (ocr, table, formula): same
         multi-task pattern as PaddleOCR-VL.

    If I'd found GLM-OCR first, I wouldn't have wasted time on TrOCR.
    Lesson learned: check the model architecture *class* before you
    check the model's benchmarks.

    Model card: https://huggingface.co/zai-org/GLM-OCR
    GitHub/SDK: https://github.com/zai-org/GLM-OCR

    The big difference from PaddleOCR in terms of code: here you pass
    the actual PIL image object inside the messages dict
    ({"type": "image", "image": img}), whereas PaddleOCR wants a
    placeholder token and the images separately.

    I followed the model card example for the message format.
    Copilot wrote _generate_and_decode. The apply_chat_template call
    here does tokenization directly (tokenize=True, return_dict=True)
    unlike PaddleOCR where we tokenize in a separate step. I don't
    love that the two backends work so differently internally but
    that's what the models expect.
    """

    def _build_messages(self, images: List[PILImage.Image], task: str):
        # One message-list per image. Each is a separate "conversation"
        # because the model processes them independently even in a batch.
        prompt = self.get_prompt(task)
        return [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            for img in images
        ]

    def _generate_and_decode(self, images, task, max_new_tokens):
        inputs = to_device(
            self.processor.apply_chat_template(
                self._build_messages(images, task),
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True,
                return_dict=True,
            )
        )
        out = generate(self.model, inputs, max_new_tokens)

        # COPILOT. This slice strips the input prompt tokens from
        # the output. "shape[-1]" is the length of the input sequence.
        # Everything after that is what the model actually generated.
        generated = out[:, inputs["input_ids"].shape[-1] :]

        return [
            t.strip()
            for t in self.processor.batch_decode(
                generated, skip_special_tokens=True
            )
        ]

    def run_single(self, image, task, max_new_tokens):
        return self._generate_and_decode([image], task, max_new_tokens)[0]

    def _run_batch_impl(self, images, task, max_new_tokens):
        return self._generate_and_decode(images, task, max_new_tokens)


# Maps the "backend" string from MODEL_REGISTRY to the actual class.
# TrOCR was going to be "trocr": TrOCRBackend here. Now it's just
# the two that actually work.
BACKEND_MAP = {"paddleocr": PaddleOCRBackend, "glmocr": GlmOCRBackend}

# -- Model loading --

# -- v2 ABANDONED: TrOCR loader --
# ABANDONED. TrOCR needs its own loading function because it uses
# different classes. This was a telling sign it wasn't going to fit.
#
# def load_trocr(model_name: str):
#     """
#     Loads TrOCR with VisionEncoderDecoderModel instead of
#     AutoModelForImageTextToText. I tried using Auto* classes first
#     and got:
#       ValueError: Unrecognized configuration class
#       <class 'transformers.models.vision_encoder_decoder
#       .configuration_vision_encoder_decoder
#       .VisionEncoderDecoderConfig'>
#       for this kind of AutoModel: AutoModelForImageTextToText.
#
#     Source for the correct loading pattern:
#     https://huggingface.co/microsoft/trocr-base-printed
#     """
#     from transformers import TrOCRProcessor, VisionEncoderDecoderModel
#     processor = TrOCRProcessor.from_pretrained(model_name)
#     model = VisionEncoderDecoderModel.from_pretrained(
#         model_name, torch_dtype=DTYPE
#     )
#     model.to(DEVICE).eval()
#     return TrOCRBackend(model_name, model, processor)


@lru_cache(maxsize=settings.duui_ocr_model_cache_size)
def load_backend(model_name: str) -> OCRBackend:
    """
    COPILOT + BORROWED. Loads a model and its processor from HuggingFace,
    wraps them in the appropriate backend class, and caches the result.

    The lru_cache decorator means we only download/load each model once.
    With cache_size=1 (default), loading a second model evicts the first.
    This is important because these models are huge and you probably can't
    fit two on one GPU.

    REVISIT. lru_cache doesn't actually free the GPU memory when it
    evicts an entry. The old model just becomes unreferenced and *eventually*
    gets garbage collected, maybe. I've seen CUDA OOM errors when switching
    models. Might need a custom cache that explicitly calls del + gc.collect()
    + torch.cuda.empty_cache() on eviction. Haven't figured out a clean
    way to do that yet.

    The AutoProcessor / AutoModelForImageTextToText pattern is from the
    HuggingFace transformers docs:
    https://huggingface.co/docs/transformers/model_doc/auto
    Copilot filled in the dtype and device placement.

    Note: this only works for models that support AutoModelForImageTextToText.
    TrOCR doesn't. That was a big part of why v2 failed. Both PaddleOCR-VL
    and GLM-OCR declare "auto_model": "AutoModelForImageTextToText" in their
    HuggingFace config, which is how the Auto* classes know what to load.
    TrOCR's config says VisionEncoderDecoderModel, which is a different
    class hierarchy entirely.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    meta = MODEL_REGISTRY[model_name]

    # v2 remnant: I had a special case here for TrOCR.
    # if meta["backend"] == "trocr":
    #     return load_trocr(model_name)

    logger.info("Loading model: %s", model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=DTYPE
    )
    # .eval() puts the model in inference mode (disables dropout etc.)
    # .to(DEVICE) moves all parameters to GPU. These two calls are in
    # every single HuggingFace example I've ever seen.
    model.to(DEVICE).eval()
    logger.info("Model loaded on %s", DEVICE)
    return BACKEND_MAP[meta["backend"]](model_name, model, processor)


# -- FastAPI --
# BORROWED. The app setup and DUUI endpoint structure is standard
# across all DUUI annotators.

app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.duui_ocr_annotator_name,
    description="Multi-model OCR Component for DUUI",
    version=settings.duui_ocr_annotator_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={"name": "TTLab Team", "url": "https://texttechnologylab.org"},
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)


# The next four endpoints are pure DUUI boilerplate. They just serve
# static content that the DUUI framework needs to discover and
# configure this annotator. Nothing interesting happens here.


@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    return Response(
        content=typesystem.to_xml().encode("utf-8"),
        media_type="application/xml",
    )


@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


@app.get("/v1/documentation")
def get_documentation():
    return TextImagerDocumentation(
        annotator_name=settings.duui_ocr_annotator_name,
        version=settings.duui_ocr_annotator_version,
        implementation_lang="Python",
        meta={
            "models": {
                name: {k: m[k] for k in ("source", "lang", "version", "tasks")}
                for name, m in MODEL_REGISTRY.items()
            },
            "supported_tasks": ALL_SUPPORTED_TASKS,
        },
        parameters={
            "model_name": "Model to use: " + ", ".join(MODEL_REGISTRY),
            "task": "OCR task: " + ", ".join(ALL_SUPPORTED_TASKS),
            "max_new_tokens": "Maximum tokens to generate",
        },
    )


@app.get("/v1/details/input_output")
def get_input_output() -> JSONResponse:
    return JSONResponse(
        content=jsonable_encoder(
            {
                "inputs": ["org.texttechnologylab.annotation.type.Image"],
                "outputs": [
                    "org.texttechnologylab.annotation.AnnotationComment"
                ],
            }
        )
    )


@app.post("/v1/process")
def post_process(request: OCRRequest):
    """
    SOLID (mostly). This is where the actual OCR happens.

    The flow:
    1. Check the requested task is valid for the chosen model
    2. Decode all images from base64/filepath to PIL
    3. Acquire the model lock (one request at a time on the GPU)
    4. Run the OCR backend on the batch
    5. Pair each result back with its original document offsets
    6. Clean up GPU memory

    I collect errors in a list instead of raising exceptions because
    DUUI expects a response even if some images failed. A partial
    result (3 out of 5 images worked) is more useful than a crash.

    FRAGILE. The finally block with cuda.empty_cache() and gc.collect()
    is my attempt at preventing memory leaks between requests. I'm not
    100% sure it's sufficient. During long runs the VRAM usage seems
    to creep up slowly. Might be a leak in the processor or in PIL.
    Haven't had time to profile it properly.
    """
    meta = MODEL_REGISTRY.get(request.model_name, {})
    ocr_results: List[OCRResult] = []
    errors: List[str] = []

    try:
        # Validate task before we do any heavy work
        supported = meta.get("tasks", [])
        if request.task not in supported:
            errors.append(
                f"Task '{request.task}' not supported by "
                f"{request.model_name}. Choose from: {supported}"
            )
        else:
            # Decode images: keep track of which ones succeeded so we
            # can match results back to the right request indices later.
            # Bad images (corrupt base64, missing files) get logged as
            # errors but don't kill the whole batch.
            pil_images, valid_indices = [], []
            for i, img_in in enumerate(request.images):
                try:
                    pil_images.append(decode_image(img_in.src))
                    valid_indices.append(i)
                except Exception as e:
                    logger.error("Failed to decode image %d: %s", i, e)
                    errors.append(f"Image {i}: {e}")

            if pil_images:
                with model_lock:
                    backend = load_backend(request.model_name)
                    texts = backend.run_batch(
                        pil_images, request.task, request.max_new_tokens
                    )

                # Pair each OCR result with the original image's
                # document offsets (begin/end). The zip with
                # valid_indices is how we skip over failed images.
                for idx, text in zip(valid_indices, texts):
                    img_in = request.images[idx]
                    ocr_results.append(
                        OCRResult(
                            text=text,
                            task=request.task,
                            begin=img_in.begin,
                            end=img_in.end,
                        )
                    )

                # Close PIL images to free memory. I kept forgetting
                # this and wondering why RAM usage kept growing.
                for img in pil_images:
                    img.close()
    except Exception as ex:
        logger.exception(ex)
        errors.append(str(ex))
    finally:
        # COPILOT. Asked "how to free GPU memory after inference in
        # pytorch" and got this. empty_cache releases unused cached
        # memory back to CUDA, gc.collect nudges Python's garbage
        # collector. Belt and suspenders.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    return OCRResponse(
        ocr_results=ocr_results,
        model_name=request.model_name,
        model_version=meta.get("version", "Unknown"),
        model_source=meta.get("source", "Unknown"),
        model_lang=meta.get("lang", "Unknown"),
        errors=errors,
        config={
            "task": request.task,
            "max_new_tokens": request.max_new_tokens,
        },
    )