[![Version](https://img.shields.io/static/v1?label=duui-transformers-topic&message=0.2.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-topic/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.41.2&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.3.0&color=red)]()

# Transformers Topic

DUUI implementation for selected Hugging-Face-based transformer [Topic tools](https://huggingface.co/models?sort=trending&search=topic) models.
## Included Models

| Name                                                                           | Revision                               | Languages |
|--------------------------------------------------------------------------------|----------------------------------------|----------|
| cardiffnlp/tweet-topic-latest-single                                           | 0ff86a9d19a5bb4045dd7ebced3714796890cfbe | EN       |
| classla/xlm-roberta-base-multilingual-text-genre-classifier                    | de7ed0ff1063e1e4bd3fd1bdda54e3ad85fb5419 | Multilingual |
| chkla/parlbert-topic-german                                                    | df343699abeb22e08c096ab3974cfd35877ce47f | DE       |
| ssharoff/genres                                                                | dc9cb7ef031abc96081d9ea96aa0e2ee1636ce04 | EN       |
 | KnutJaegersberg/topic-classification-IPTC-subject-labels                       | fe1fb726c12850b1e2f6ed3fa379a0a6c4558a4c | Multilingual |
 | poltextlab/xlm-roberta-large-manifesto-cap                                     | 5f19b49c412d504c1c8357a31367a65c0302717e | Multilingual |
| manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1  | 06c046795a3b7b9822755f0a73776f8fabec3977 | Multilingual |
 
# How To Use

For using duui-transformers-topic as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run -p 9714:9714 docker.texttechnologylab.org/duui-transformers-topic:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-topic/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-topic:latest")
        .withParameter("model_name", "cardiffnlp/tweet-topic-latest-single")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
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

@misc{Bagci:2024,
  author         = {Bagci, Mevlüt},
  title          = {Hugging-Face-based topic models as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-topic}
}

```
