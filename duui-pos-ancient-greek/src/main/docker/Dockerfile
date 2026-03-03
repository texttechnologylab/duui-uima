# Builds the container image for the Ancient Greek POS tagger.
# Uses a fine-tuned XLM-RoBERTa model (qbnguyen/ancient-greek-pos-xlmr)
# to tag parts of speech in Ancient Greek text, served as a DUUI
# annotator component.


# BORROWED. The overall skeleton (WORKDIR, EXPOSE, pip pattern,
# ARG/ENV pairs, uvicorn entrypoint) is lifted from existing
# DUUI annotator Dockerfiles. Specifically I had these open while
# writing this:
#   - duui-sentencizer-spacy (the spaCy one with all the model downloads)
#   - duui-flair-pos (the Flair POS tagger, for the ARG/ENV pattern)
#   - my own duui-ocr Dockerfile (for the HF model pre-download trick)
# All from: https://github.com/texttechnologylab/DockerUnifiedUIMAInterface
#
# Last meaningful edit: Feb 2026


# BORROWED. python:3.10-slim instead of the full 3.10 that other DUUI
# normally uses. ChatGPT suggested this when I complained about image
# size. The "-slim" variant strips out a lot of system tooling we
# don't need (man pages, extra locales, etc). I stuck with 3.10
# specifically because that's what DUUI often have on.
FROM python:3.10-slim

WORKDIR /usr/src/app

EXPOSE 9714

# -- System dependencies --
# FRAGILE. This block exists because python:3.10-slim doesn't include
# a C compiler, and at least one of our pip dependencies (I think it's
# a tokenizers thing? or maybe sentencepiece?) needs to compile from
# source.
#
# I only found this out because `pip install` was failing with a
# cryptic "error: command 'gcc' not found" buried in the output.
# Asked ChatGPT what was going on and it explained that slim images
# don't ship build tools.
#
# The --no-install-recommends flag and the rm -rf cleanup at the end
# are from ChatGPT's suggestion to keep the image small. The idea is:
# install gcc and friends, let pip use them to compile whatever it
# needs, and the compiled .so files stay even though we could
# theoretically remove build-essential afterward. I haven't bothered
# with that cleanup because multi-stage builds seem like a rabbit hole
# I don't need right now.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# -- Python dependencies --
# SOLID. Same pattern as every other DUUI Dockerfile.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -- Model pre-download --
# BORROWED (the pattern) from my OCR Dockerfile, which I originally
# got from Copilot. AI is eating itself ouroboros style I guess.
#
# Downloads the fine-tuned XLM-RoBERTa model for Ancient Greek POS
# tagging at *build time* so the container doesn't need internet
# access at runtime. This is the same trick the OCR Dockerfile uses
# for PaddleOCR-VL and GLM-OCR.
#
# The model itself is from me on Hugging Face. It's XLM-R
# fine-tuned on Ancient Greek POS data. I self-trained it and
# upload to Hugging Face instead of using locally, fitting DUUI
# framework so far.
#
# Note: unlike the OCR models which use AutoModelForImageTextToText,
# this one uses AutoModelForTokenClassification because POS tagging
# is a token classification task.
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForTokenClassification; \
AutoModelForTokenClassification.from_pretrained('qbnguyen/ancient-greek-pos-xlmr'); \
AutoTokenizer.from_pretrained('qbnguyen/ancient-greek-pos-xlmr')"

# -- Source files --
# Just listing the source files, nothing more to comment.
COPY src/main/python/duui_pos_ancient_greek.py .
COPY src/main/python/duui_pos_ancient_greek.lua .
COPY src/main/python/TypeSystemPOS.xml .

# -- Configuration --
# BORROWED. The ENV pattern is from every other DUUI Dockerfile.
# The other Dockerfiles use the ARG/ENV pair pattern (ARG with a
# default, then ENV=$ARG) so you can override at build time with
# --build-arg. I simplified to just ENV here because I don't
# actually need build-time overrides yet. I'm the only one building
# this. If this ever goes into TTLab's pipeline I should
# probably switch to the ARG/ENV pair pattern to match the others.
#
# REVISIT: should DUUI_POS_AG_MODEL_PATH be configurable? Right now
# it's hardcoded to the one model. But if someone fine-tunes a better
# Ancient Greek POS model later, being able to swap it via env var
# without rebuilding the image would be nice. Leaving it as ENV for
# that reason even though currently there's only one option.
ENV DUUI_POS_AG_ANNOTATOR_NAME="duui-pos-ancient-greek"
ENV DUUI_POS_AG_ANNOTATOR_VERSION="0.1.0"
ENV DUUI_POS_AG_LOG_LEVEL="DEBUG"
ENV DUUI_POS_AG_MODEL_PATH="qbnguyen/ancient-greek-pos-xlmr"

# -- Startup --
# BORROWED. Identical pattern to every other DUUI annotator.
# uvicorn runs the FastAPI app on port 9714.
#
# Not setting --workers here (the other Dockerfiles put it in CMD).
# I should probably add CMD ["--workers", "1"] to match the
# convention, but it defaults to 1 anyway so it's not breaking
# anything. In the OCR Dockerfile I made a note about multiple workers
# each loading their own copy of the model into VRAM. Same concern
# applies here, so 1 worker is correct for now.
ENTRYPOINT ["uvicorn", "duui_pos_ancient_greek:app", \
            "--host", "0.0.0.0", "--port", "9714"]