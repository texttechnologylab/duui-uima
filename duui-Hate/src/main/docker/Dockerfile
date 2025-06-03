FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Andrazp/multilingual-hate-speech-robacofi')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='alexandrainst/da-hatespeech-detection-base')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='l3cube-pune/me-hate-roberta')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='GroNLP/hateBERT')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-hate-latest')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-english')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='deepset/bert-base-german-cased-hatespeech-GermEval18Coarse')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-german')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-spanish')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-polish')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-portugese')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-italian')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-arabic')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-french')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/dehatebert-mono-indonesian')"
#RUN python -c "from nubia_score import Nubia; nubia = Nubia()"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='facebook/roberta-hate-speech-dynabench-r4-target')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='irlab-udc/MetaHateBERT')"

#COPY ./src/main/python/HateBERT_hateval ./HateBERT_hateval

RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('EZiisk/EZ_finetune_Vidgen_model_RHS_Best'); AutoTokenizer.from_pretrained('EZiisk/EZ_finetune_Vidgen_model_RHS_Best_Tokenizer')"

# service script
COPY ./src/main/python/TypeSystemHate.xml ./TypeSystemHate.xml
COPY ./src/main/python/evaluator.py ./evaluator.py
COPY ./src/main/python/hatechecker.py ./hatechecker.py
COPY ./src/main/python/duui_hate.lua ./duui_hate.lua
COPY ./src/main/python/duui_hate.py ./duui_hate.py


# meta data
ARG ANNOTATOR_NAME="duui-hate:app"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

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

# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE



ENTRYPOINT ["uvicorn", "duui_hate:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

