FROM python:3.10

WORKDIR /usr/src/app

EXPOSE 9714

# dependencies
RUN pip install setuptools wheel
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

RUN #python -m spacy download en_core_web_lg

RUN #python -c "import nltk; nltk.download('all', download_dir='nltk_data')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='02shanky/finetuned-twitter-xlm-roberta-base-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='DReAMy-lib/xlm-roberta-large-DreamBank-emotion-presence')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='MilaNLProc/xlm-emo-t')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='michellejieli/emotion_text_classifier')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='finiteautomata/bertweet-base-emotion-analysis')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ActivationAI/distilbert-base-uncased-finetuned-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='mrm8488/t5-base-finetuned-emotion')"
#RUN python -c "from transformers import pipeline; pipeline(model='microsoft/mdeberta-v3-base')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='pranaydeeps/EXALT-Baseline')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='boltuix/bert-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='MilaNLProc/feel-it-italian-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cardiffnlp/twitter-roberta-base-emotion-multilabel-latest')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='finiteautomata/beto-emotion-analysis')"
#RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('poltextlab/xlm-roberta-large-pooled-MORES', token='<TOKEN>'); AutoTokenizer.from_pretrained('lytang/MiniCheck-Flan-T5-Large')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='daveni/twitter-xlm-roberta-emotion-es')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='ChrisLalk/German-Emotions')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='msgfrom96/xlm_emo_multi')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='cointegrated/rubert-tiny2-cedr-emotion-detection')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Aniemore/rubert-tiny2-russian-emotion-detection')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Johnson8187/Chinese-Emotion-Small')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Johnson8187/Chinese-Emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='AnasAlokla/multilingual_go_emotions')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Zoopa/emotion-classification-model')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='esuriddick/distilbert-base-uncased-finetuned-emotion')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='Panda0116/emotion-classification-model')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='lordtt13/emo-mobilebert')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='alex-shvets/roberta-large-emopillars-contextual-emocontext')"
#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='AdapterHub/bert-base-uncased-pf-emo')"
RUN python -c "from pytorch_transformers import (BertTokenizer, BertModel, BertConfig,); BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False); BertModel.from_pretrained('bert-base-multilingual-cased')"


# copy scripts
COPY ./src/main/python/TypeSystemEmotion.xml ./TypeSystemEmotion.xml
COPY ./src/main/python/Emo_mDeBERTa2.py ./Emo_mDeBERTa2.py
COPY ./src/main/python/EmotionDetection.py ./EmotionDetection.py
COPY ./src/main/python/duui_transformers_emotion.py ./duui_transformers_emotion.py
COPY ./src/main/python/duui_emotion.lua ./duui_emotion.lua
COPY ./src/main/python/script.py ./script.py
COPY ./src/main/python/utils.py ./utils.py
#COPY ./src/main/python/UniversalJoy/models-multilingual/small-all.pt ./UniversalJoy/models-multilingual/small-all.pt
#COPY ./src/main/python/UniversalJoy/models-multilingual/large-all.pt ./UniversalJoy/models-multilingual/large-all.pt
#COPY ./src/main/python/UniversalJoy/models-multilingual/combi-all.pt ./UniversalJoy/models-multilingual/combi-all.pt
#COPY ./src/main/python/UniversalJoy/models-huge-english/huge-en.pt ./UniversalJoy/models-huge-english/huge-en.pt
#COPY ./src/main/python/UniversalJoy/models-large-monolingual/large-en.pt ./UniversalJoy/models-large-monolingual/large-en.pt
#COPY ./src/main/python/UniversalJoy/models-large-monolingual/large-es.pt ./UniversalJoy/models-large-monolingual/large-es.pt
#COPY ./src/main/python/UniversalJoy/models-large-monolingual/large-pt.pt ./UniversalJoy/models-large-monolingual/large-pt.pt
#COPY ./src/main/python/UniversalJoy/models-small-monolingual/small-en.pt ./UniversalJoy/models-small-monolingual/small-en.pt
#COPY ./src/main/python/UniversalJoy/models-small-monolingual/small-es.pt ./UniversalJoy/models-small-monolingual/small-es.pt
#COPY ./src/main/python/UniversalJoy/models-small-monolingual/small-pt.pt ./UniversalJoy/models-small-monolingual/small-pt.pt
#COPY ./src/main/python/UniversalJoy/models-small-monolingual/small-tl.pt ./UniversalJoy/models-small-monolingual/small-tl.pt
COPY ./src/main/python/UniversalJoy/models-small-monolingual/small-zh.pt ./UniversalJoy/models-small-monolingual/small-zh.pt
#COPY ./src/main/python/nltk_data/ ./nltk_data/
#COPY ./src/main/python/pol_emo_mDeBERTa/ ./pol_emo_mDeBERTa/

# log level
ARG LOG_LEVEL="DEBUG"
ENV LOG_LEVEL=$LOG_LEVEL

# config
ARG MODEL_CACHE_SIZE=3
ENV MODEL_CACHE_SIZE=$MODEL_CACHE_SIZE

# meta data
ARG ANNOTATOR_NAME="duui-transformers-emotion"
ENV ANNOTATOR_NAME=$ANNOTATOR_NAME
ARG ANNOTATOR_VERSION="unset"
ENV ANNOTATOR_VERSION=$ANNOTATOR_VERSION

# Model Info
ARG MODEL_VERSION=0.1
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_NAME=""
ENV MODEL_NAME=$MODEL_NAME
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG

# offline mode for huggingface
ARG DUUI_TRANSFORMERS_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$DUUI_TRANSFORMERS_TRANSFORMERS_OFFLINE




ENTRYPOINT ["uvicorn", "duui_transformers_emotion:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]