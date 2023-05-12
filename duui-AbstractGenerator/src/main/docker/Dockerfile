FROM python:3.8

WORKDIR /usr/src/app

RUN pip install --upgrade pip
RUN pip install setuptools wheel
COPY requirements.txt ./requirements.txt
RUN pip install --no-deps -r ./requirements.txt


# vars
#ARG MODEL="en_ner_eco_md"
##ARG MODEL="en_ner_eco_biobert"
#ENV MODEL=$MODEL
#
#ARG LINKING="gbif_backbone"
#ENV LINKING=$LINKING

# service script
COPY ./python/dkpro-core-types.xml ./dkpro-core-types.xml
COPY ./python/communication.lua ./communication.lua
COPY ./python/duui_abstractgenerator.py ./duui_abstractgenerator.py

ENTRYPOINT ["uvicorn", "duui_abstractgenerator:app", "--host", "0.0.0.0", "--port", "9714"]
CMD ["--workers", "1"]