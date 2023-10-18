FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install transformers sentence-transformers
RUN pip install numpy scipy scikit-learn
RUN pip install symspellpy fastapi uvicorn[standard] dkpro-cassis
RUN pip install --ignore-installed pydantic-settings==2.0.2


# meta data
ARG TEXTIMAGER_SPBERT_ANNOTATOR_NAME="textimager_duui_spellcheck:app"
ENV TEXTIMAGER_SPBERT_ANNOTATOR_NAME=$TEXTIMAGER_SPBERT_ANNOTATOR_NAME
ARG TEXTIMAGER_SPBERT_ANNOTATOR_VERSION="unset"
ENV TEXTIMAGER_SPBERT_ANNOTATOR_VERSION=$TEXTIMAGER_SPBERT_ANNOTATOR_VERSION

# log level
ARG TEXTIMAGER_SPBERT_LOG_LEVEL="DEBUG"
ENV TEXTIMAGER_SPBERT_LOG_LEVEL=$TEXTIMAGER_SPBERT_LOG_LEVEL

# config
ARG TEXTIMAGER_SPBERT_MODEL_CACHE_SIZE=3
ENV TEXTIMAGER_SPBERT_MODEL_CACHE_SIZE=$TEXTIMAGER_SPBERT_MODEL_CACHE_SIZE

# Model Info
ARG TEXTIMAGER_SPBERT_MODEL_NAME="spbert"
ENV TEXTIMAGER_SPBERT_MODEL_NAME=$TEXTIMAGER_SPBERT_MODEL_NAME
ARG TEXTIMAGER_SPBERT_MODEL_VERSION=0.1
ENV TEXTIMAGER_SPBERT_MODEL_VERSION=$TEXTIMAGER_SPBERT_MODEL_VERSION

# service script
COPY ./src/main/python/TypeSystemSPBERT.xml ./TypeSystemSPBERT.xml
COPY ./src/main/python/de-100k.txt ./de-100k.txt
COPY ./src/main/python/BERT_converter.py ./BERT_converter.py
COPY ./src/main/python/cos_sim.py ./cos_sim.py
COPY ./src/main/python/MASK_BERT.py ./MASK_BERT.py
COPY ./src/main/python/spellchecker.py ./spellchecker.py
COPY ./src/main/python/sp_correction.py ./sp_correction.py
COPY ./src/main/python/textimager_duui_spbert.lua ./textimager_duui_spbert.lua
COPY ./src/main/python/textimager_duui_spbert.py ./textimager_duui_spbert.py
COPY ./requirements.txt ./requirements.txt



ENTRYPOINT ["uvicorn", "textimager_duui_spbert:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

