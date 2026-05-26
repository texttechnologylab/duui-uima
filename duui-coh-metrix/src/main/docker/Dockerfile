FROM python:3.11

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('omw-1.4')"

ADD https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/raw/main/mrc_psycholinguistic_database.csv ./src/main/resources/mrc_psycholinguistic_database.csv

COPY ./src/main/resources/word_frequencies_de_dewiki-20220301-sample10000.csv ./src/main/resources/word_frequencies_de_dewiki-20220301-sample10000.csv
COPY ./src/main/resources/word_frequencies_en_enwiki-20220301-sample10000.csv ./src/main/resources/word_frequencies_en_enwiki-20220301-sample10000.csv

ARG DUUI_COH_METRIX_LOG_LEVEL="DEBUG"
ENV DUUI_COH_METRIX_LOG_LEVEL=$DUUI_COH_METRIX_LOG_LEVEL

ARG DUUI_COH_METRIX_ANNOTATOR_NAME="duui-coh-metrix"
ENV DUUI_COH_METRIX_ANNOTATOR_NAME=$DUUI_COH_METRIX_ANNOTATOR_NAME
ARG DUUI_COH_METRIX_ANNOTATOR_VERSION="unset"
ENV DUUI_COH_METRIX_ANNOTATOR_VERSION=$DUUI_COH_METRIX_ANNOTATOR_VERSION

VOLUME ["/usr/src/app/src/main/resources/germanet"]

ARG DUUI_COH_METRIX_GERMANET_PATH="/usr/src/app/src/main/resources/germanet"
ENV DUUI_COH_METRIX_GERMANET_PATH=$DUUI_COH_METRIX_GERMANET_PATH

COPY ./src/main/resources/TypeSystem.xml ./src/main/resources/TypeSystem.xml
COPY ./src/main/python/duui.py ./src/main/python/duui.py
COPY ./src/main/lua/communication.lua ./src/main/lua/communication.lua

ENTRYPOINT ["uvicorn", "src.main.python.duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
