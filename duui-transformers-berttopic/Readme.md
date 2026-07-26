[![Version](https://img.shields.io/static/v1?label=duui-transformers-berttopic&message=1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-topic/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.41.2&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.3.0&color=red)]()
[![Version](https://img.shields.io/static/v1?label=BERTopic&message=0.16.4&color=purple)]()

# Transformers BERTopic

DUUI implementation for a trained [BERTopic](https://github.com/MaartenGr/BERTopic) model. 

# How To Use

For using duui-transformers-berttopic as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).


## Start Docker container

```
## CPU container
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-transformers-bertopic:[version]

## CUDA container
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-transformers-berttopic-cuda:latest
```

Find all available image tags here: [https://docker.texttechnologylab.org/v2/duui-transformers-bertopic/tags/list](https://docker.texttechnologylab.org/v2/duui-transformers-bertopic/tags/list)

## Versions

| Tag       | Type System                                                                                                    | Model                                                                                      | Training Data                                                                 |
|-----------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| `0.1`     | [BertTopic](https://github.com/texttechnologylab/UIMATypeSystem/blob/uima-3/src/main/resources/desc/type/TypeSystemBertTopic.xml) (specific to BERTopic) | [BERTopic_Wikipedia](https://huggingface.co/MaartenGr/BERTopic_Wikipedia)                 | [Wikipedia pages ](https://huggingface.co/datasets/Cohere/wikipedia-22-12)                                                               |
| `latest` / `1.0` | [UnifiedTopic](https://github.com/texttechnologylab/UIMATypeSystem/blob/uima-3/src/main/resources/desc/type/TypeSystemUnifiedTopic.xml) (generic, works for different models) | [BERTopic_ML-ArXiv-Abstracts](https://huggingface.co/b-verma/BERTopic_ML-ArXiv-Abstracts) | [ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers)   |

## Run within DUUI

```
composer.add(
    new DUUIRemoteDriver.Component("docker.texttechnologylab.org/duui-transformers-bertopic:latest")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
);
```

### Parameters

| Name | Description  |
| ---- |--------------|
| `selection`  | Segmentation type to be used for the selection of text segments in the input text |

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

@misc{Bagci:2024,
  author         = {Verma, Bhuvanesh},
  title          = {BERTopic based topic models as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-berttopic}
}

```
