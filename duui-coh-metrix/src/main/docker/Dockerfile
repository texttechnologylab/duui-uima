FROM python:3.14.6

WORKDIR /usr/src/app

EXPOSE 9714

RUN pip install setuptools wheel

COPY ./requirements.txt ./requirements.txt
RUN pip install --no-deps -r requirements.txt

# WordNet 3.0 and the CMU Pronouncing Dictionary (English indices and syllables)
RUN python -c "import nltk; nltk.download('wordnet', download_dir='/usr/src/app/src/main/resources/nltk_data')"
RUN python -c "import nltk; nltk.download('omw-1.4', download_dir='/usr/src/app/src/main/resources/nltk_data')"
RUN python -c "import nltk; nltk.download('cmudict', download_dir='/usr/src/app/src/main/resources/nltk_data')"

ADD https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database/raw/main/mrc_psycholinguistic_database.csv ./src/main/resources/mrc_psycholinguistic_database.csv

# Bundled resources (licences: src/main/resources/LICENSES.md)
COPY ./src/main/resources/word_frequencies_de_dewiki-20220301-sample10000.csv ./src/main/resources/word_frequencies_de_dewiki-20220301-sample10000.csv
COPY ./src/main/resources/word_frequencies_en_enwiki-20220301-sample10000.csv ./src/main/resources/word_frequencies_en_enwiki-20220301-sample10000.csv
COPY ./src/main/resources/mrc_psycholinguistic_database_de.csv ./src/main/resources/mrc_psycholinguistic_database_de.csv
COPY ./src/main/resources/gum_weights_en.csv ./src/main/resources/gum_weights_en.csv
COPY ./src/main/resources/easability/text_easability_de_refv1-ud-rr-cr3.json ./src/main/resources/easability/text_easability_de_refv1-ud-rr-cr3.json
COPY ./src/main/resources/easability/text_easability_en_refa-en-rr-gn-cr3.json ./src/main/resources/easability/text_easability_en_refa-en-rr-gn-cr3.json
COPY ./src/main/resources/LICENSES.md ./src/main/resources/LICENSES.md

ARG DUUI_COH_METRIX_LOG_LEVEL="DEBUG"
ENV DUUI_COH_METRIX_LOG_LEVEL=$DUUI_COH_METRIX_LOG_LEVEL

ARG DUUI_COH_METRIX_ANNOTATOR_NAME="duui-coh-metrix"
ENV DUUI_COH_METRIX_ANNOTATOR_NAME=$DUUI_COH_METRIX_ANNOTATOR_NAME
ARG DUUI_COH_METRIX_ANNOTATOR_VERSION="unset"
ENV DUUI_COH_METRIX_ANNOTATOR_VERSION=$DUUI_COH_METRIX_ANNOTATOR_VERSION

# Resources the image does not contain (licensed, non-commercial or without a licence; README, "Resources supplied by the
# user"): GermaNet in its own volume, the other files in one volume under these names. A missing file leaves its indices None
# (and, through the Text Easability inputs, the PC* components).
VOLUME ["/usr/src/app/src/main/resources/germanet"]
ENV DUUI_COH_METRIX_GERMANET_PATH="/usr/src/app/src/main/resources/germanet"
VOLUME ["/usr/src/app/src/main/resources/external"]
ENV DUUI_COH_METRIX_DIMLEX_PATH="/usr/src/app/src/main/resources/external/DimLex.xml"
ENV DUUI_COH_METRIX_DIMLEX_EN_PATH="/usr/src/app/src/main/resources/external/en_dimlex.xml"
ENV DUUI_COH_METRIX_KOEPER_PATH="/usr/src/app/src/main/resources/external/affective_norms.txt.gz"
ENV DUUI_COH_METRIX_BIRCHENOUGH_PATH="/usr/src/app/src/main/resources/external/13428_2016_718_MOESM1_ESM.csv"
ENV DUUI_COH_METRIX_SUBTLEX_DE_PATH="/usr/src/app/src/main/resources/external/SUBTLEXde_SUBTLWF.csv"
ENV DUUI_COH_METRIX_SUBTLEX_US_PATH="/usr/src/app/src/main/resources/external/SUBTLEXus74286wordstextversion.txt"
ENV DUUI_COH_METRIX_BRYSBAERT_PATH="/usr/src/app/src/main/resources/external/brysbaert_concreteness.csv"
ENV DUUI_COH_METRIX_KUPERMAN_PATH="/usr/src/app/src/main/resources/external/kuperman_aoa.csv"

ENV DUUI_COH_METRIX_LSA_USE_TRUNCATED_SVD=false
ENV DUUI_COH_METRIX_LSA_SVD_COMPONENTS=100

COPY ./src/main/resources/TypeSystem.xml ./src/main/resources/TypeSystem.xml
COPY ./src/main/python/cohmetrix_core.py ./src/main/python/cohmetrix_core.py
COPY ./src/main/python/duui.py ./src/main/python/duui.py
COPY ./src/main/lua/communication.lua ./src/main/lua/communication.lua

ENTRYPOINT ["uvicorn", "src.main.python.duui:app", "--host", "0.0.0.0", "--port" ,"9714"]
CMD ["--workers", "1"]
