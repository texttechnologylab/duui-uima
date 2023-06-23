FROM nvidia/cuda:11.0.3-base-ubuntu20.04

RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install --no-install-recommends -y build-essential software-properties-common curl && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /usr/src/app

EXPOSE 9714

# ---------------------------------------------------------
# config
# ---------------------------------------------------------
# model names options:
# 1) "tuner007/pegasus_paraphrase" : PegasusBase
# 2) "humarin/chatgpt_paraphraser_on_T5_base" : T5Base
# 3) "eugenesiow/bart-paraphrase": BartBase
# 4) "prithivida/parrot_paraphraser_on_T5": ParrotBase
ARG MODEL_NAME="humarin/chatgpt_paraphraser_on_T5_base"
ENV MODEL_NAME=$MODEL_NAME
# use cuda
ARG CUDA=1
ENV CUDA=$CUDA
# gpu_id
ARG GPU_ID=0
ENV GPU_ID=$GPU_ID
# ---------------------------------------------------------
# ---------------------------------------------------------
# meta data
# ---------------------------------------------------------
# name
ARG TEXTIMAGER_PARA_ANNOTATOR_NAME="textimager-duui-paraphraser"
ENV TEXTIMAGER_PARA_ANNOTATOR_NAME=$TEXTIMAGER_PARA_ANNOTATOR_NAME
# version
ARG TEXTIMAGER_PARA_ANNOTATOR_VERSION=0.1
ENV TEXTIMAGER_PARA_ANNOTATOR_VERSION=$TEXTIMAGER_PARA_ANNOTATOR_VERSION

# service script
COPY ./src/main/python/communication.lua ./communication.lua
COPY ./src/main/python/dkpro-core-types.xml ./dkpro-core-types.xml
COPY ./src/main/python/duui_paraphraser.py ./duui_paraphraser.py
COPY ./src/main/python/paraphraser.py ./paraphraser.py
COPY ./src/main/python/server.sh ./server.sh
COPY ./src/main/python/requirements.txt ./requirements.txt

# requirements
RUN python3 -m pip install -r requirements.txt

# load models
# parrot
RUN python -c "from parrot import Parrot; model = Parrot('prithivida/parrot_paraphraser_on_T5')"
# pegasus
RUN python -c "from transformers import PegasusTokenizer; tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_paraphrase')"
RUN python -c "from transformers import PegasusForConditionalGeneration; model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_paraphrase')"
# bart
RUN python -c "from transformers import BartTokenizer; tokenizer = BartTokenizer.from_pretrained('eugenesiow/bart-paraphrase')"
RUN python -c "from transformers import BartForConditionalGeneration; model = BartForConditionalGeneration.from_pretrained('eugenesiow/bart-paraphrase')"
# t5
RUN python -c "from transformers import AutoTokenizer; tokenizer = AutoTokenizer.from_pretrained('humarin/chatgpt_paraphraser_on_T5_base')"
RUN python -c "from transformers import AutoModelForSeq2SeqLM; model = AutoModelForSeq2SeqLM.from_pretrained('humarin/chatgpt_paraphraser_on_T5_base')"


ENTRYPOINT ["uvicorn", "duui_paraphraser:app", "--host", "0.0.0.0", "--port", "9714"]
#ENTRYPOINT ["python3", "-m", "uvicorn", "duui_paraphraser:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]