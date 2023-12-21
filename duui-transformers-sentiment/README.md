[![Version](https://img.shields.io/static/v1?label=duui-transformers-sentiment&message=0.1.2&color=blue)](https://docker.texttechnologylab.org/v2/textimager-duui-transformers-sentiment/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.8&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.21.1&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=1.11.0&color=red)]()

# Transformers Sentiment

DUUI implementation for selected Hugging-Face-based transformer [sentiment tools](https://huggingface.co/models?sort=trending&search=sentiment).

## Included Models

| Name                                    | Revision                                 | Languages |
| --------------------------------------- | ---------------------------------------- | --------- |
| cardiffnlp/twitter-roberta-base-sentiment | b636d90b2ed53d7ba6006cefd76f29cd354dd9da | EN      |
| cardiffnlp/twitter-roberta-base-sentiment-latest | 5916057ce88cf0a408a195082b6c06d3dce12552 | EN |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | f3e34b6c30bf27b6649f72eca85d0bbe79df1e55 | AR, EN, FR, DE, HI, IT, SP, PT |
| clampert/multilingual-sentiment-covid19 | eea3f8e26d2828dbf9f0f1d939dd868396ec863c | EN, FR, DE |
| cmarkea/distilcamembert-base-sentiment  | b7804e295dc3cf2aa8ce8cff83f22e0bdd249558 | FR        |
| finiteautomata/bertweet-base-sentiment-analysis | cf6b0f60e84096e077c171fe3176093674370291 | EN |
| j-hartmann/sentiment-roberta-large-english-3-classes | f995433eb6d79d26702ab9335bfde472a9933ee4 | EN |
| LiYuan/amazon-review-sentiment-analysis | 0aacda6423e43213da4e50a0f30cfcdb42a5c725 | EN, DE, FR, ES, IT, NL |
| mdraw/german-news-sentiment-bert        | 7b4abebe1c3fcfbc62dc0435e480807a80c18210 | DE        |
| nlptown/bert-base-multilingual-uncased-sentiment | e06857fdb0325a7798a8fc361b417dfeec3a3b98 | EN, DE, FR, ES, IT, NL |
| oliverguhr/german-sentiment-bert        | c5c8dd0c5b966460dce1b7c5851bd90af1d2c6b6 | DE |
| philschmid/distilbert-base-multilingual-cased-sentiment-2 | 83ff874f93aacbba79642abfe2a274a3c874232b | EN, DE, FR, ES, ZH, JA |
| siebert/sentiment-roberta-large-english | 6eac71655a474ee4d6d0eee7fa532300c537856d | EN        |

# How To Use

For using duui-transformers-sentiment as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/textimager-duui-transformers-sentiment:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/textimager-duui-transformers-sentiment/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/textimager-duui-transformers-sentiment:latest")
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

