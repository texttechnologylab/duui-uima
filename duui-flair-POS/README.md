[![Version](https://img.shields.io/static/v1?label=python&message=3.10&color=blue)]()
[![Version](https://img.shields.io/static/v1?label=pytorch&message=1.13.1&color=orange)]()
[![Version](https://img.shields.io/static/v1?label=flair&message=0.12&color=orange)]()
[![Version](https://img.shields.io/static/v1?label=cuda&message=11.7.1&color=green)]()

# Flair POS

A DUUI pipeline for the use of [Flair](https://github.com/flairNLP/flair) for POS tagging.

# HowToUse

For using taxoNERD as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Use as Stand-Alone-Image

```sh
docker run docker.texttechnologylab.org/flair/pos:latest
```

## Run with a specific port

```sh
docker run -p 1000:9714 docker.texttechnologylab.org/flair/pos:latest
```

## Run within DUUI

```java
composer.add(new DUUIDockerDriver.
    Component("docker.texttechnologylab.org/flair/pos:latest")
    .withScale(iWorkers)
    .withImageFetching());
```

## Supported Models

The models can be chosen using `language` parameter in the [LUA communication layer](./src/main/lua/communication_layer.lua) by the keys from the table below.

| Key            | Name              | Type                    | Language           | Dataset                    |          Performance | Contributor / Notes                 |
| -------------- | ----------------- | ----------------------- | ------------------ | -------------------------- | -------------------: | ----------------------------------- |
| `en`           | `pos`             | POS-tagging             | English            | Ontonotes                  |     98.19 (Accuracy) |                                     |
| `en-fast`      | `pos-fast`        | POS-tagging             | English            | Ontonotes                  |      98.1 (Accuracy) | (fast model)                        |
| `en-upos`      | `upos`            | POS-tagging (universal) | English            | Ontonotes                  |      98.6 (Accuracy) |                                     |
| `en-upos-fast` | `upos-fast`       | POS-tagging (universal) | English            | Ontonotes                  |     98.47 (Accuracy) | (fast model)                        |
| `multi`        | `pos-multi`       | POS-tagging             | Multilingual       | UD Treebanks               | 96.41 (average acc.) | (12 languages)                      |
| `multi-fast`   | `pos-multi-fast`  | POS-tagging             | Multilingual       | UD Treebanks               | 92.88 (average acc.) | (12 languages)                      |
| `ar`           | `ar-pos`          | POS-tagging             | Arabic (+dialects) | combination of corpora     |                      |                                     |
| `de`           | `de-pos`          | POS-tagging             | German             | UD German - HDT            |     98.50 (Accuracy) |                                     |
| `de-twitter`   | `de-pos-tweets`   | POS-tagging             | German             | German Tweets              |     93.06 (Accuracy) | stefan-it                           |
| `da`           | `da-pos`          | POS-tagging             | Danish             | Danish Dependency Treebank |                      | AmaliePauli                         |
| `ms`           | `ml-pos`          | POS-tagging             | Malayalam          | 30000 Malayalam sentences  |                   83 | sabiqueqb                           |
| `ms-upos`      | `ml-upos`         | POS-tagging             | Malayalam          | 30000 Malayalam sentences  |                   87 | sabiqueqb                           |
| `pt`           | `pt-pos-clinical` | POS-tagging             | Portuguese         | PUCPR                      |                92.39 | LucasFerroHAILab for clinical texts |
| `uk`           | `pos-ukrainian`   | POS-tagging             | Ukrainian          | Ukrainian UD               |           97.93 (F1) | dchaplinsky                         |

*See: <https://flairnlp.github.io/docs/tutorial-basics/part-of-speech-tagging>*

## Environment Arguments

The following environment arguments can be set to change the behavior of Flair:

- `MODEL_CACHE_SIZE`: determines the number of Flair models that will remain loaded in memory at any given time.
- `FLAIR_BATCH_SIZE`: determines the batch size during inference.

### Default Values

```sh
MODEL_CACHE_SIZE=1
FLAIR_BATCH_SIZE=128
```

# Cite

If you want to use the DUUI image please quote this as follows:

- Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]
- Manuel Stoeckel. (2022) "Flair as DUUI-Component for POS Tagging." [[LINK](https://github.com/texttechnologylab/duui-uima/tree/main/duui-flair-POS)]

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
               Docker Unified UIMA Interface (DUUI), a scalable, flexible, lightweight,
               and feature-rich framework for automatic distributed analysis
               of text corpora that leverages Big Data experience and virtualization
               with Docker. We evaluate DUUI{'}s communication approach against
               a state-of-the-art approach and demonstrate its outstanding behavior
               in terms of time efficiency, enabling the analysis of big text
               data.}
}

@misc{Stoeckel:2022:DUUI:Flair:POS,
  author         = {Stoeckel, Manuel},
  title          = {Flair as DUUI-Component for POS Tagging},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-flair-POS}
}
```
