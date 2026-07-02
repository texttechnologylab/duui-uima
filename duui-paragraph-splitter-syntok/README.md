# Sentence Splitting and Paragraph Detection with syntok

DUUI component for rule-based sentence splitting and paragraph detection using [syntok](https://github.com/fnl/syntok). Takes the raw document text directly from the CAS with no prior annotations required, segments it into sentences and paragraphs, and writes `Sentence` and/or `Paragraph` annotations back. Both annotation types can be toggled independently at runtime.

# How To Use

For using duui-paragraph-splitter-syntok as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-paragraph-splitter-syntok:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-paragraph-splitter-syntok/tags/list

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-paragraph-splitter-syntok:latest")
        .withParameter("write_paragraphs", "true")
        .withParameter("write_sentences", "true")
);
```

### Parameters

| Name | Description | Default |
|---|---|---|
| `write_paragraphs` | Write `Paragraph` annotations to the CAS | `true` |
| `write_sentences` | Write `Sentence` annotations to the CAS | `false` |

### Input types

No UIMA annotations are required as input. The component reads only the raw document text and language from the CAS.

| Source | Description |
|---|---|
| CAS document text | Full plain text of the document passed directly to syntok |
| CAS document language | Language code (`en`, `de`, `es`) forwarded in the request |

### Output types

| Type | Description |
|---|---|
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph` | One annotation per detected paragraph with begin/end character offsets. Written only when `write_paragraphs` is `true`. |
| `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence` | One annotation per detected sentence with begin/end character offsets. Written only when `write_sentences` is `true`. |

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

@misc{duui-paragraph-splitter-syntok,
  author         = {Baumartz, Daniel},
  title          = {Sentence Splitting and Paragraph Detection via syntok as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-paragraph-splitter-syntok}
}

```
