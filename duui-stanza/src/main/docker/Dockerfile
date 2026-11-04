FROM python:3.11-slim

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

# torch (with CUDA) in its own layer, the same as in duui-wtpsplit and duui-chonky, so that the images share it
RUN pip install --no-cache-dir torch==2.14.1

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements.txt

# Stanza models (package default_accurate, the processors the component runs) downloaded at build time; creating each
# pipeline once (REUSE_RESOURCES: no new Stanza download) also downloads its transformer from Hugging Face. The container
# then runs offline (download_method=None in duui.py, HF_HUB_OFFLINE).
ENV STANZA_RESOURCES_DIR=/usr/src/app/stanza_resources
RUN python -c "import stanza; \
[stanza.download(lang, package='default_accurate', processors='tokenize,pos,lemma,depparse', model_dir='$STANZA_RESOURCES_DIR') for lang in ('de', 'en')]; \
[stanza.Pipeline(lang, processors='tokenize,pos,lemma,depparse', package='default_accurate', tokenize_pretokenized=True, download_method=stanza.DownloadMethod.REUSE_RESOURCES, dir='$STANZA_RESOURCES_DIR', use_gpu=False) for lang in ('de', 'en')]"
ENV HF_HUB_OFFLINE=1

ARG DUUI_STANZA_LOG_LEVEL="INFO"
ENV DUUI_STANZA_LOG_LEVEL=$DUUI_STANZA_LOG_LEVEL

ARG DUUI_STANZA_ANNOTATOR_NAME="duui-stanza"
ENV DUUI_STANZA_ANNOTATOR_NAME=$DUUI_STANZA_ANNOTATOR_NAME
ARG DUUI_STANZA_ANNOTATOR_VERSION="unset"
ENV DUUI_STANZA_ANNOTATOR_VERSION=$DUUI_STANZA_ANNOTATOR_VERSION

COPY ./src/main/resources/TypeSystem.xml ./TypeSystem.xml
COPY ./src/main/python/duui.py ./duui.py
COPY ./src/main/lua/communication.lua ./communication.lua

ENTRYPOINT ["uvicorn", "duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
