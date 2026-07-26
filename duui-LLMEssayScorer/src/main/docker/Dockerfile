FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt


#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('KevSun/Engessay_grading_ML'); AutoTokenizer.from_pretrained('KevSun/Engessay_grading_ML')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('JacobLinCool/IELTS_essay_scoring_safetensors'); AutoTokenizer.from_pretrained('JacobLinCool/IELTS_essay_scoring_safetensors')"

#Begrading Models runs on different Server. The Information are given by the user with the parameters by the service call.

#LLMAESSlowScorer Models runs on different Server. The Information are given by the user with the parameters by the service call.

# service script
COPY ./src/main/python/TypeSystemEssayScorer.xml ./TypeSystemEssayScorer.xml
COPY ./src/main/python/duui_essayscorer.lua ./duui_essayscorer.lua
COPY ./src/main/python/duui_essayscorer.py ./duui_essayscorer.py
COPY ./src/main/python/EssayScorer.py ./EssayScorer.py
COPY ./src/main/python/BeGradingScorer.py ./BeGradingScorer.py
COPY ./src/main/python/LLMAESDataset.py ./LLMAESDataset.py
COPY ./src/main/python/LLMAESScorer.py ./LLMAESScorer.py
COPY ./src/main/python/AAGScorer.py ./AAGScorer.py
COPY ./src/main/python/GradingMedicalEducation.py ./GradingMedicalEducation.py
COPY ./src/main/python/ASAP_template_mts.xlsx ./ASAP_template_mts.xlsx
COPY ./src/main/python/toefel11_template_mts.xlsx ./toefel11_template_mts.xlsx
COPY ./src/main/python/ASAP_template_vanilla.xlsx ./ASAP_template_vanilla.xlsx
COPY ./src/main/python/toefel11_template_vanilla.xlsx ./toefel11_template_vanilla.xlsx
COPY ./src/main/python/AESMTSScorer.py ./AESMTSScorer.py
COPY ./src/main/python/GradeLikeHuman.py ./GradeLikeHuman.py
COPY ./src/main/python/essay_meta_data.py ./essay_meta_data.py
COPY ./src/main/python/EssayEvalScoring.py ./EssayEvalScoring.py
COPY ./src/main/python/llm_functions.py ./llm_functions.py
COPY ./src/main/python/prompts.py ./prompts.py
COPY ./src/main/python/run_text.py ./run_text.py



# meta data
ARG ANNOTATOR_NAME="duui_essayscorer:app"
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
ARG MODEL_SPEC_NAME=""
ENV MODEL_SPEC_NAME=$MODEL_SPEC_NAME
ARG MODEL_VERSION=""
ENV MODEL_VERSION=$MODEL_VERSION
ARG MODEL_SOURCE=""
ENV MODEL_SOURCE=$MODEL_SOURCE
ARG MODEL_LANG=""
ENV MODEL_LANG=$MODEL_LANG



# offline mode for huggingface
ARG TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=$TEXTIMAGER_DUUI_TRANSFORMERS_SENTIMENT_TRANSFORMERS_OFFLINE


ENTRYPOINT ["uvicorn", "duui_essayscorer:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

