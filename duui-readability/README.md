# Text Readability Analysis

DUUI component for computing standard readability metrics using [py-readability-metrics](https://github.com/cdimascio/py-readability-metrics). Takes the raw document text directly from the CAS with no prior annotations required, computes nine classical readability formulas in a single pass, and writes one `CategoryCoveredTagged` annotation per metric spanning the full document. Each annotation carries the numeric score and a JSON `tags` field with formula-specific detail.

# How To Use

For using duui-readability as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-readability:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-readability/tags/list

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-readability:latest")
);
```

### Parameters

This component requires no runtime parameters. All nine readability metrics are always computed.

### Input types

No UIMA annotations are required as input. The component reads only the raw document text and language from the CAS.

| Source | Description |
|---|---|
| CAS document text | Full plain text |
| CAS document language | Forwarded in the request payload; not used to select metrics |

### Output types

| Type | Description |
|---|---|
| `org.hucompute.textimager.uima.type.category.CategoryCoveredTagged` | One annotation per readability metric, spanning the full document. `value` holds the metric name, `score` holds the numeric result, and `tags` holds a JSON string with formula-specific detail fields (see metric table above). |
| `org.texttechnologylab.annotation.AnnotatorMetaData` | Created for each `CategoryCoveredTagged` annotation. Records the annotator name, version, model name (`py-readability-metrics`), and model version. |
| `org.texttechnologylab.annotation.DocumentModification` | Single annotation recording the annotator name, processing timestamp, and a version comment. |

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

@misc{duui-readability,
  author         = {Baumartz, Daniel},
  title          = {Text Readability Analysis as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-readability}
}

```
