# Coh-Metrix Text Cohesion Analysis [WIP]

DUUI implementation of coh-metrix, computing text cohesion and readability indices over pre-annotated UIMA documents. Produces 100+ numerical indices across descriptive, referential cohesion, LSA, lexical diversity, syntactic complexity, connective, and situation model categories.

## Prerequisites

This annotator requires a **pre-annotated UIMA CAS** with the following annotation types (produced by e.g. the DUUI spaCy component):

- `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph`
- `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence`
- `org.texttechnologylab.uima.type.spacy.SpacyToken`
- `org.texttechnologylab.uima.type.spacy.SpacyNounChunk`

Word vectors on tokens are required for LSA indices (`LSA*`). Without them, LSA indices will return zero vectors.

# How To Use

For using duui-coh-metrix as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-coh-metrix:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-coh-metrix/tags/list

```
docker run --rm -p 1000:9714 \
  -v /path/to/germanet:/usr/src/app/src/main/resources/germanet \
  docker.texttechnologylab.org/v2/duui-coh-metrix:latest
```

Without GermaNet, German polysemy, hypernymy, and causal/intentional verb metrics fall back to seed-word lists or return `-1`.

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
);
```

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

@misc{duui-coh-metrix,
  author         = {Baumartz, Daniel},
  title          = {Coh-Metrix Text Cohesion Analysis as {DUUI} component},
  year           = {2026},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-coh-metrix}
}

```
