FROM python:3.12

WORKDIR /usr/src/app

EXPOSE 9714

COPY ./reqiurements.txt ./reqiurements.txt
RUN pip install -r reqiurements.txt
COPY ./src/main/python/LLMDetection.py ./LLMDetection.py


#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('TrustSafeAI/RADAR-Vicuna-7B'); AutoTokenizer.from_pretrained('TrustSafeAI/RADAR-Vicuna-7B')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta'); AutoTokenizer.from_pretrained('Hello-SimpleAI/chatgpt-detector-roberta')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('MayZhou/e5-small-lora-ai-generated-detector'); AutoTokenizer.from_pretrained('MayZhou/e5-small-lora-ai-generated-detector')"

#RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('tiiuae/Falcon3-1B-Base'); AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Base'); AutoModelForCausalLM.from_pretrained('tiiuae/Falcon3-1B-Instruct'); AutoTokenizer.from_pretrained('tiiuae/Falcon3-1B-Instruct');"

#RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2'); AutoTokenizer.from_pretrained('gpt2');"

#RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B'); AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B'); AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125m'); AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m');"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('GeorgeDrayson/modernbert-ai-detection'); AutoTokenizer.from_pretrained('GeorgeDrayson/modernbert-ai-detection')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('yuchuantian/AIGC_detector_env2'); AutoTokenizer.from_pretrained('yuchuantian/AIGC_detector_env2')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('yuchuantian/AIGC_detector_zhv2'); AutoTokenizer.from_pretrained('yuchuantian/AIGC_detector_zhv2')"

#RUN python -c "from transformers import AutoTokenizer; from LLMDetection import RobertaClassifier; RobertaClassifier.from_pretrained('SuperAnnotate/ai-detector'); AutoTokenizer.from_pretrained('SuperAnnotate/ai-detector')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('fakespot-ai/roberta-base-ai-text-detection-v1'); AutoTokenizer.from_pretrained('fakespot-ai/roberta-base-ai-text-detection-v1')"

#RUN python -c "from transformers import AutoTokenizer; from LLMDetection import DesklibAIDetectionModel; DesklibAIDetectionModel.from_pretrained('desklib/ai-text-detector-v1.01'); AutoTokenizer.from_pretrained('desklib/ai-text-detector-v1.01')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('yaful/MAGE'); AutoTokenizer.from_pretrained('yaful/MAGE')"

RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('FacebookAI/xlm-roberta-base'); AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('VSAsteroid/ai-text-detector-hc3'); AutoTokenizer.from_pretrained('VSAsteroid/ai-text-detector-hc3')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('SJTU-CL/RoBERTa-large-ArguGPT-sent'); AutoTokenizer.from_pretrained('SJTU-CL/RoBERTa-large-ArguGPT-sent')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('SJTU-CL/RoBERTa-large-ArguGPT'); AutoTokenizer.from_pretrained('SJTU-CL/RoBERTa-large-ArguGPT')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('raj-tomar001/LLM-DetectAIve_deberta-base'); AutoTokenizer.from_pretrained('raj-tomar001/LLM-DetectAIve_deberta-base')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('wangkevin02/AI_Detect_Model'); AutoTokenizer.from_pretrained('wangkevin02/AI_Detect_Model')"

#RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('openai-community/gpt2-medium'); AutoTokenizer.from_pretrained('openai-community/gpt2-medium')"

#RUN python -c "from transformers import pipeline; pipeline('text-classification', model='nealcly/detection-longformer', tokenizer='nealcly/detection-longformer', max_length=512, truncation=True, padding=True)"

#RUN python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; T5ForConditionalGeneration.from_pretrained('google-t5/t5-small', return_dict=True); T5Tokenizer.from_pretrained('google-t5/t5-small')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('openai-community/roberta-large-openai-detector'); AutoTokenizer.from_pretrained('openai-community/roberta-large-openai-detector')"

#RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoModelForSequenceClassification.from_pretrained('PirateXX/AI-Content-Detector'); AutoTokenizer.from_pretrained('PirateXX/AI-Content-Detector')"


# service script
COPY ./src/main/python/TypeSystemLLMDetection.xml ./TypeSystemLLMDetection.xml
COPY ./src/main/python/duui_llmdetection.lua ./duui_llmdetection.lua
COPY ./src/main/python/duui_llmdetection.py ./duui_llmdetection.py
COPY ./src/main/python/PHDScore.py ./PHDScore.py
COPY ./src/main/python/InstrinsicDim.py ./InstrinsicDim.py

#COPY ./src/main/python/T5Sentinel.0613.pt ./T5Sentinel.0613.pt

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

