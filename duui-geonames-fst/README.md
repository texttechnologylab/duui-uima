[![Version](https://img.shields.io/static/v1?label=duui-geonames-fst-de&message=0.3.3&color=blue)](https://docker.texttechnologylab.org/v2/duui-geonames-fst/de/tags/list)
[![Version](https://img.shields.io/static/v1?label=duui-geonames-fst-eu&message=0.3.3&color=blue)](https://docker.texttechnologylab.org/v2/duui-geonames-fst/eu/tags/list)
[![Version](https://img.shields.io/static/v1?label=duui-geonames-fst-europe&message=0.3.3&color=blue)](https://docker.texttechnologylab.org/v2/duui-geonames-fst/europe/tags/list)
[![Version](https://img.shields.io/static/v1?label=duui-geonames-fst-europe-central&message=0.3.3&color=blue)](https://docker.texttechnologylab.org/v2/duui-geonames-fst/europe-central/tags/list)
[![Version](https://img.shields.io/static/v1?label=DUUI&message=compatible&color=green)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface)

# GeoNames FST DUUI

DUUI implementation for GeoNames-based location linking using finite-state transducers. The component reads location annotations from a configured DUUI source view and writes resolved `org.texttechnologylab.annotation.geonames.GeoNamesEntity` annotations to the configured DUUI target view.

## Included Images

| Name | Docker image | Description |
|--------------------------------------------------------------|--------------------------------------------------------------|----------------------------------------|
| de | `docker.texttechnologylab.org/duui-geonames-fst/de:latest` | GeoNames lookup for Germany |
| eu | `docker.texttechnologylab.org/duui-geonames-fst/eu:latest` | GeoNames lookup for the European Union |
| europe | `docker.texttechnologylab.org/duui-geonames-fst/europe:latest` | GeoNames lookup for Europe |
| europe-central | `docker.texttechnologylab.org/duui-geonames-fst/europe-central:latest` | GeoNames lookup for Central Europe |
| ------------------------------------------------------------ |--------------------------------------------------------------|----------------------------------------|

# How To Use

For using `duui-geonames-fst` as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```bash
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-geonames-fst/europe:latest
```

Find all available image tags here:

- https://docker.texttechnologylab.org/v2/duui-geonames-fst/de/tags/list
- https://docker.texttechnologylab.org/v2/duui-geonames-fst/eu/tags/list
- https://docker.texttechnologylab.org/v2/duui-geonames-fst/europe/tags/list
- https://docker.texttechnologylab.org/v2/duui-geonames-fst/europe-central/tags/list

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component(
            "docker.texttechnologylab.org/duui-geonames-fst/europe:latest")
        .withScale(iWorkers)
        .withParameter("timeout", "5000")
        .withParameter("mode", "levenshtein")
        .withParameter("max_dist", "2")
        .withParameter("min_length", "5")
        .withParameter("result_selection", "first")
        .withSourceView("roberta-ner-multilingual")
        .withTargetView("geonames-roberta-ner-multilingual")
        .withImageFetching()
        .build()
);
```

## Parameters

| Parameter | Default | Description |
|--------------------------------------------------------------|------------------------------------------|----------------------------------------|
| `mode` | `find` | Lookup mode. Supported modes include `find` and `levenshtein`. |
| `max_dist` | - | Maximum edit distance for `levenshtein` lookup. |
| `state_limit` | - | Optional state limit for `levenshtein` lookup. |
| `min_length` | - | Minimum query length. |
| `result_selection` | `first` | Result selection strategy. |
| `timeout` | - | Request timeout in milliseconds. |
| `annotation_type` | `de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location` | Source annotation type read from the configured source view. |
| ------------------------------------------------------------ |------------------------------------------|----------------------------------------|

## Input and Output

The component expects location annotations in the configured DUUI source view:

```java
.withSourceView("roberta-ner-multilingual")
```

Resolved GeoNames annotations are written to the configured DUUI target view:

```java
.withTargetView("geonames-roberta-ner-multilingual")
```

The default input annotation type is:

```text
de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location
```

The output annotation type is:

```text
org.texttechnologylab.annotation.geonames.GeoNamesEntity
```

## Build Docker images

Build all variants locally:

```bash
./build-all.sh 0.3.3
```

Build and push all variants:

```bash
docker login docker.texttechnologylab.org
PUSH=true ./build-all.sh 0.3.3
```

# Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399.
[[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

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
  pages     = {385--399},
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf},
  abstract  = {Automatic analysis of large corpora is a complex task, especially
               in terms of time efficiency. This complexity is increased by the
               fact that flexible, extensible text analysis requires the continuous
               integration of ever new tools. Since there are no adequate frameworks
               for these purposes in the field of NLP, and especially in the
               context of UIMA, that are not outdated or unusable for security
               reasons, we present a new approach to address the latter task:
               Docker Unified UIMA Interface (DUUI), a scalable, flexible,
               lightweight, and feature-rich framework for automatic distributed
               analysis of text corpora that leverages Big Data experience and
               virtualization with Docker. We evaluate DUUI{'}s communication
               approach against a state-of-the-art approach and demonstrate its
               outstanding behavior in terms of time efficiency, enabling the
               analysis of big text data.}
}

@misc{Bagci:2026,
  author       = {Bagci, Mevlüt},
  title        = {GeoNames FST as {DUUI} component},
  year         = {2026},
  howpublished = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-geonames-fst}
}
```