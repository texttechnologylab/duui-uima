FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/bert-base-uncased-hatexplain')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='worldbank/naija-xlm-twitter-base-hate')"

#COPY ./src/main/python/HateBERT_abuseval ./HateBERT_abuseval

#COPY ./src/main/python/HateBERT_offenseval ./HateBERT_offenseval

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='pysentimiento/bertweet-hate-speech')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='pysentimiento/robertuito-hate-speech')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='pysentimiento/bertabaporu-pt-hate-speech')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='pysentimiento/bert-it-hate-speech')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='IMSyPP/hate_speech_multilingual')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='IMSyPP/hate_speech_en')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='IMSyPP/hate_speech_it')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='IMSyPP/hate_speech_nl')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='IMSyPP/hate_speech_slo')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-hate-multiclass-latest')"

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-large-sensitive-multilabel')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('EZiisk/EZ_finetune_Vidgen_model_RHS_Best'); AutoTokenizer.from_pretrained('EZiisk/EZ_finetune_Vidgen_model_RHS_Best_Tokenizer')"

# service script
COPY ./src/main/python/TypeSystemOffensive.xml ./TypeSystemOffensive.xml
COPY ./src/main/python/Offensivechecker.py ./Offensivechecker.py
COPY ./src/main/python/duui_offensive.lua ./duui_offensive.lua
COPY ./src/main/python/duui_offensive.py ./duui_offensive.py


# meta data
ARG ANNOTATOR_NAME="duui-offensive:app"
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



ENTRYPOINT ["uvicorn", "duui_offensive:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

