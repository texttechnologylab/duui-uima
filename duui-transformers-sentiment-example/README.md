[![Version](https://img.shields.io/static/v1?label=duui-transformers-sentiment-example&message=1.0.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-sentiment/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.8&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.21.1&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=1.11.0&color=red)]()

NOTE: This is a simplified version of [duui-transformers-sentiment](https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-sentiment) for the DUUI developer manual.

# Transformers Sentiment

DUUI implementation for selected Hugging-Face-based transformer [sentiment tools](https://huggingface.co/models?sort=trending&search=sentiment).

## Included Models

| Name                                    | Revision                                 | Languages |
| --------------------------------------- | ---------------------------------------- | --------- |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | f3e34b6c30bf27b6649f72eca85d0bbe79df1e55 | AR, EN, FR, DE, HI, IT, SP, PT |

# How To Use

For using duui-transformers-sentiment-example as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-transformers-sentiment-example:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-sentiment-example/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-sentiment-example:latest")
        .withScale(iWorkers)
        .withImageFetching()
);
```

### Parameters

| Name | Description |
| ---- | ----------- |
| `model_name` | Model to use, see table above |
| `selection`  | Use `text` to process the full document text or any selectable UIMA type class name |

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

@misc{Baumartz:2022,
  author         = {Baumartz, Daniel},
  title          = {Hugging-Face-based sentiment models as DUUI component},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-sentiment}
}

```

