[![Version](https://img.shields.io/static/v1?label=duui-time&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-time/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()
[![Version](https://img.shields.io/static/v1?label=FastAPI&message=0.115%2B&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=UIMA&message=TimeX3&color=red)]()

# DUUI Time Detection

DUUI implementation for temporal expression detection and TimeX3 annotation.

The component detects temporal expressions in selected UIMA annotations or in the full document text and writes ISO-TimeML-compatible `TimeX3` annotations into the CAS. The implementation supports multiple backends. Each Docker image is built for exactly one model/backend and one language configuration.

## Included Models

| Name | Backend | Model / Resource | Languages | Notes |
| ---- | ------- | ---------------- | --------- | ----- |
| `microsoft` | Microsoft Recognizers-Text | `recognizers-text-suite==1.0.2a2` | multilingual | Rule-based temporal recognition. |
| `duckling` | Duckling HTTP service | external Duckling server | multilingual | Requires a running Duckling container or service. |
| `sutime` | Stanford CoreNLP SUTime HTTP service | external CoreNLP/SUTime server | multilingual | Requires a running CoreNLP server. |
| `german-gelectra` | Hugging Face token classification | `satyaalmasian/temporal_tagger_German_GELECTRA` | DE | German temporal tagger. |
| `bert-got-a-date` | Hugging Face token classification | `satyaalmasian/temporal_tagger_BERT_tokenclassifier` | EN | English temporal tagger. |
| `hf-token-classification` | Hugging Face token classification | custom `MODEL_SPECNAME` | configurable | Generic Hugging Face token-classification backend. |
| `tei2go-de` | spaCy / TEI2GO | `de_tei2go` | DE | One image per language. |
| `tei2go-en` | spaCy / TEI2GO | `en_tei2go` | EN | One image per language. |
| `tei2go-es` | spaCy / TEI2GO | `es_tei2go` | ES | One image per language. |
| `tei2go-fr` | spaCy / TEI2GO | `fr_tei2go` | FR | One image per language. |
| `tei2go-it` | spaCy / TEI2GO | `it_tei2go` | IT | One image per language. |
| `tei2go-pt` | spaCy / TEI2GO | `pt_tei2go` | PT | One image per language. |
| `timexy-de` | spaCy / Timexy | `de_core_news_sm` | DE | One image per language. |
| `timexy-en` | spaCy / Timexy | `en_core_web_sm` | EN | One image per language. |
| `timexy-fr` | spaCy / Timexy | `fr_core_news_sm` | FR | One image per language. |

## Build Images

The build script creates one Docker image per model and language.

Build one model:

```bash
./docker_build.sh microsoft de
```

Build all Timexy language variants:

```bash
./docker_build.sh timexy all
```

Build all default images:

```bash
./docker_build.sh all
```

Build a custom Hugging Face token-classification model:

```bash
./docker_build.sh hf-token-classification de satyaalmasian/temporal_tagger_German_GELECTRA
```

## Start Docker Container

Run a DUUI Time Detection image locally:

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-time-[modelname]-[lang]:latest
```

Example:

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-time-microsoft-de:latest
```

TEI2GO example:

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-time-tei2go-de:latest
```

Timexy example:

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-time-timexy-de:latest
```

## External Services

### Duckling

Start Duckling:

```bash
docker run --rm -p 8000:8000 rasa/duckling
```

Start the DUUI Time Duckling wrapper:

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-time-duckling-de:latest
```

In DUUI, pass the Duckling URL as runtime parameter:

```java
.withParameter("duckling_url", "http://127.0.0.1:8000")
.withParameter("duckling_timezone", "Europe/Berlin")
```

If DUUI runs inside another Docker container, use the reachable host name, for example:

```java
.withParameter("duckling_url", "http://host.docker.internal:8000")
```

### SUTime / CoreNLP

Start CoreNLP:

```bash
docker run --rm -p 9000:9000 --name corenlp nlpbox/corenlp
```

Start the DUUI Time SUTime wrapper:

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-time-sutime-de:latest
```

In DUUI, pass the CoreNLP URL as runtime parameter:

```java
.withParameter("corenlp_url", "http://127.0.0.1:9000")
```

If DUUI runs inside another Docker container, use the reachable host name, for example:

```java
.withParameter("corenlp_url", "http://host.docker.internal:9000")
```

## Run within DUUI

For using DUUI Time Detection as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Docker Driver

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-time-microsoft-de:latest")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        .withParameter("document_creation_time", "2026-06-09")
);
```

### Remote Driver

If the container or local Python service is already running on port `9714`:

```java
composer.add(
    new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        .withParameter("document_creation_time", "2026-06-09")
);
```

### Duckling Remote Driver

```java
composer.add(
    new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        .withParameter("document_creation_time", "2026-06-09")
        .withParameter("duckling_url", "http://127.0.0.1:8000")
        .withParameter("duckling_timezone", "Europe/Berlin")
);
```

### SUTime Remote Driver

```java
composer.add(
    new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        .withParameter("document_creation_time", "2026-06-09")
        .withParameter("corenlp_url", "http://127.0.0.1:9000")
);
```

## Parameters

| Name | Description |
| ---- | ----------- |
| `selection` | Use `text` to process the full document text or any selectable UIMA type class name, for example `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence`. |
| `document_creation_time` | Reference date for relative temporal expressions, for example `2026-06-09`. |
| `threshold` | Optional confidence threshold for Hugging Face token-classification models. |
| `batch_size` | Optional batch size for Hugging Face token-classification models. |
| `duckling_url` | Runtime URL of the Duckling HTTP service. Required for `MODEL_NAME=duckling`. |
| `duckling_timezone` | Runtime timezone for Duckling normalization, for example `Europe/Berlin`. |
| `corenlp_url` | Runtime URL of the CoreNLP/SUTime HTTP service. Required for `MODEL_NAME=sutime`. |

## Local Development

Start the service locally without Docker:

```bash
export ANNOTATOR_NAME="duui-time"
export ANNOTATOR_VERSION="0.1.0"
export LOG_LEVEL="DEBUG"

export MODEL_NAME="microsoft"
export MODEL_SPECNAME="recognizers-text-suite"
export MODEL_VERSION="1.0.2a2"
export MODEL_SOURCE="https://github.com/microsoft/Recognizers-Text"
export MODEL_LANG="de"
export MODEL_CACHE_SIZE="1"

uvicorn duui_time:app --host 0.0.0.0 --port 9714 --workers 1
```

## Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

## BibTeX

```bibtex
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  editor    = {Bouamor, Houda and Pino, Juan and Bali, Kalika},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399}
}

@misc{Bagci:2026,
  author       = {Bagci, Mevlüt},
  title        = {Temporal expression detection models as {DUUI} component},
  year         = {2026},
  howpublished = {https://github.com/texttechnologylab/duui-uima}
}
```