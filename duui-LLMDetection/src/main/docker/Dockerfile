FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt


#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('TrustSafeAI/RADAR-Vicuna-7B'); AutoTokenizer.from_pretrained('TrustSafeAI/RADAR-Vicuna-7B')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta'); AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('MayZhou/e5-small-lora-ai-generated-detector'); AutoTokenizer.from_pretrained('MayZhou/e5-small-lora-ai-generated-detector')"

#RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('tiiuae/Falcon3-1B-Base'); AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Base'); AutoModelForCausalLM.from_pretrained('tiiuae/Falcon3-1B-Instruct'); AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Instruct');"

#RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2'); AutoTokenizer.from_pretrained('gpt2');"

RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2'); AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('tiiuae/Falcon3-1B-Instruct'); AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Instruct');"

# service script
COPY ./src/main/python/TypeSystemLLMDetection.xml ./TypeSystemLLMDetection.xml
COPY ./src/main/python/LLMDetection.py ./LLMDetection.py
COPY ./src/main/python/duui_llmdetection.lua ./duui_llmdetection.lua
COPY ./src/main/python/duui_llmdetection.py ./duui_llmdetection.py


# meta data
ARG ANNOTATOR_NAME="duui_llmdetection:app"
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


ENTRYPOINT ["uvicorn", "duui_llmdetection:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]

