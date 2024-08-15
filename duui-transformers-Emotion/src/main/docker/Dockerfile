FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714


# copy scripts
COPY ./src/main/python/TypeSystemEmotion.xml ./TypeSystemEmotion.xml
COPY ./src/main/python/pol_emo_mDeBERTa/ ./pol_emo_mDeBERTa/
COPY ./src/main/python/Emo_mDeBERTa2.py ./Emo_mDeBERTa2.py
COPY ./src/main/python/EmotionDetection.py ./EmotionDetection.py
COPY ./src/main/python/duui_transformers_emotion.py ./duui_transformers_emotion.py
COPY ./src/main/python/duui_emotion.lua ./duui_emotion.lua
COPY ./src/main/python/nltk_data/ ./nltk_data/

# dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_lg

RUN python -c "from transformers import pipeline; pipeline('text-classification', model='02shanky/finetuned-twitter-xlm-roberta-base-emotion')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='MilaNLProc/xlm-emo-t')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='michellejieli/emotion_text_classifier')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-emotion')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='finiteautomata/bertweet-base-emotion-analysis')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ActivationAI/distilbert-base-uncased-finetuned-emotion')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')"
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='mrm8488/t5-base-finetuned-emotion')"

# log level
ARG EMOTION_LOG_LEVEL="DEBUG"
ENV EMOTION_LOG_LEVEL=$EMOTION_LOG_LEVEL

# config
ARG EMOTION_MODEL_CACHE_SIZE=3
ENV EMOTION_MODEL_CACHE_SIZE=$EMOTION_MODEL_CACHE_SIZE

# meta data
ARG EMOTION_ANNOTATOR_NAME="duui-transformers-emotion"
ENV EMOTION_ANNOTATOR_NAME=$EMOTION_ANNOTATOR_NAME
ARG EMOTION_ANNOTATOR_VERSION="unset"
ENV EMOTION_ANNOTATOR_VERSION=$EMOTION_ANNOTATOR_VERSION

# Model Info
ARG EMOTION_MODEL_VERSION=0.1
ENV EMOTION_MODEL_VERSION=$EMOTION_MODEL_VERSION

# offline mode for huggingface
#ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
#ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE




ENTRYPOINT ["uvicorn", "duui_transformers_emotion:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
