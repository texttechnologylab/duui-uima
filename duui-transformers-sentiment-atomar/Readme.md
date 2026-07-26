[![Version](https://img.shields.io/static/v1?label=duui-transformers-sentiment-atomar&message=0.3.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-sentiment/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.41.2&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.3.0&color=red)]()

# Transformers Sentiment

DUUI implementation for selected Hugging-Face-based transformer [Sentiment tools](https://huggingface.co/models?sort=trending&search=sentiment) models.
## Included Models

| Name                                                          |                                                                                            | Revision                           | Languages |
|---------------------------------------------------------------|--------------------------------------------------------------------------------------------|------------------------------------|------|
| twitter-xlm-roberta-base-sentiment                            | https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment | f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8 | Multilingual |
| citizenlab-twitter-xlm-roberta-base-sentiment-finetunned      | https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned | a9381f1d9e6f8aac74155964c2f6ea9a63a9e9a6 | Multilingual |
| distilbert-base-multilingual-cased-sentiments-student         | https://huggingface.co/lxyuan/distilbert-base-multilingual-cased-sentiments-student | cf991100d706c13c0a080c097134c05b7f436c45 | Multilingual |
| philschmid-distilbert-base-multilingual-cased-sentiments-student | https://huggingface.co/philschmid/distilbert-base-multilingual-cased-sentiment | b45a713783e49ac09c94dfda4bff847f4ad771c5 | Multilingual |
| cardiffnlp-sentiment-en                                       | https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest | 4ba3d4463bd152c9e4abd892b50844f30c646708 | EN   |
| roberta-based-en | https://huggingface.co/j-hartmann/sentiment-roberta-large-english-3-classes | 81cdc0fe3eee1bc18d95ffdfb56b2151a39c9007 | EN   |
| finance-sentiment-de  | https://huggingface.co/bardsai/finance-sentiment-de-base | 51b3d03f716eaa093dc42130f675839675a07b9a  | DE   |
| german-sentiment-bert  | https://huggingface.co/oliverguhr/german-sentiment-bert | b1177ff59e305c966836ba2825d3dc2efc53f125  | DE   |

# How To Use

For using duui-transformers-sentiment as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-transformers-sentiment-atomar-[modelname]:latest

```

Find all available image tags here: [https://docker.texttechnologylab.org/v2/duui-transformers-sentiment-atomar-[modelname]/tags/list](https://docker.texttechnologylab.org/v2/duui-transformers-topic-[modelname]/tags/list)

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-sentiment-atomar-[modelname]:latest")
        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
);
```

### Parameters

| Name | Description |
| ---- | ----------- |
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
  title          = {Hugging-Face-based Sentiment models as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-sentiment-atomar}
}

```
