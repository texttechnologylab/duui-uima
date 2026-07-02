# Lemmatization with HanTa

DUUI component for morphological lemmatization using [HanTa (Hannover Tagger)](https://github.com/wartaal/HanTa), a probabilistic morphological tagger for German. Accepts pre-tokenized and pre-sentence-split text via `Token` and `Sentence` annotations in the CAS, runs HanTa's morphological analysis per sentence, and writes `Lemma` annotations back into the CAS.

## Prerequisites

The component requires pre-existing `Token` and `Sentence` annotations in the CAS before it is invoked.

# How To Use

For using duui-hanta as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-hanta:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-hanta/tags/list

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-hanta:latest")
);
```

### Input types

The CAS must contain sentence and token segmentation before running this component:

| Type | Description |
|---|---|
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence` | Sentence boundaries used to group tokens for HanTa |
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token` | Individual tokens covered by each sentence; their `coveredText` is passed to HanTa |

### Output types

| Type | Description |
|---|---|
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma` | One annotation per input token. Spans the same character offsets as the source token and stores the morphological lemma in `value`. |

# Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

## BibTeX

```
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

@misc{duui-hanta,
  author         = {Konca, Maxim},
  title          = {Lemmatization via {HanTa} as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-hanta}
}

```
