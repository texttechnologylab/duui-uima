FROM python:3.8

WORKDIR /usr/src/app

EXPOSE 9714


# meta data
ARG ANNOTATOR_NAME="duui-Factchecking:app"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config
ARG MODEL_CACHE_SIZE=3
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE

# config
ARG MODEL_CACHE_SIZE=1
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_VERSION=""
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG



COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt
RUN python -m nltk.downloader punkt
#RUN python -c "from evaluator import get_evaluator; get_evaluator('fact', device='cpu')"
#RUN python -c "from nubia_score import Nubia; nubia = Nubia()"

#RUN python -c "from transformers import BertForSequenceClassification, BertTokenizer; BertForSequenceClassification.from_pretrained('manueldeprada/FactCC'); BertTokenizer.from_pretrained('manueldeprada/FactCC')"
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; AutoModelForSeq2SeqLM.from_pretrained('lytang/MiniCheck-Flan-T5-Large'); AutoTokenizer.from_pretrained('lytang/MiniCheck-Flan-T5-Large')"

# service script
COPY ./src/main/python/TypeSystemFactChecking.xml ./TypeSystemFactChecking.xml
COPY ./src/main/python/scorer.py ./scorer.py
COPY ./src/main/python/evaluator.py ./evaluator.py
COPY ./src/main/python/utils.py ./utils.py
COPY ./src/main/python/factchecker.py ./factchecker.py
COPY ./src/main/python/duui_fact.lua ./duui_fact.lua
COPY ./src/main/python/duui_fact.py ./duui_fact.py


#RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118


COPY ./src/main/python/minicheck.py ./minicheck.py
COPY ./src/main/python/inference.py ./inference.py

# offline mode for huggingface
ARG DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE

ENTRYPOINT ["uvicorn", "duui_fact:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

