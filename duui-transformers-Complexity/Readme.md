[![Version](https://img.shields.io/static/v1?label=duui-transformers-complexity&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-complexity/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.38.1color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Transformers Complexity

DUUI implementation for selected Hugging-Face-based transformer [Complexity tools](https://huggingface.co/models?sort=trending&search=fill-mask) as a DUUI component.
The embeddings will be use to compute the complexity of the text. The complexities are the following metrics: euclidean,cosine,wasserstein,distance,jensenshannon,bhattacharyya.
## Included Models

| Name                               | Revision                                  | Languages    |
|------------------------------------|-------------------------------------------|--------------|
| intfloat/multilingual-e5-base      | d13f1b27baf31030b7fd040960d60d909913633f  | Multilingual |
| google-bert/bert-base-multilingual-cased | 3f076fdb1ab68d5b2880cb87a0886f315b8146f8  | Multilingual |
| FacebookAI/xlm-roberta-large       | c23d21b0620b635a76227c604d44e43a9f0ee389  | Multilingual |
| cardiffnlp/twitter-xlm-roberta-base | 4c365f1490cb329b52150ad72f922ea467b5f4e6  | Multilingual |
| facebook/xlm-v-base                | 068c75dd7733d2640b3a98114e3e94196dc543fe1 |  Multilingual  |
| setu4993/LEALLA-small              | 8fadf81fe3979f373ba9922ab616468a4184b266  |  Multilingual |
| sentence-transformers/LaBSE        | 5513ed8dd44a9878c7d4fe8646d4dd9df2836b7b  |  Multilingual    |
| Twitter/twhin-bert-large           | 2786782c0f659550e3492093e4aab963d495243 |  Multilingual  |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 543dcf585e1eb6d4ece18c2e0c29474d9c5146b70  |  Multilingual |
| sentence-transformers/distiluse-base-multilingual-cased-v2  | 501a2afbd9deb9f028b175cc6060f38bb5055ce4  | Multilingual |
# How To Use

For using duui-transformers-complexity as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-transformers-complexity:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-complexity/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-complexity:latest")
        .withParameter("model_name", model)
        .withParameter("model_art", "bert")
        .withParameter("complexity_compute", complexities)
        .withParameter("embeddings_keep", "1")
);
```

### Parameters

| Name      | Description                                                                                               |
|-----------|-----------------------------------------------------------------------------------------------------------|
| `model_name` | Model to use, see table above                                                                             |
| `model_art` | Bert for BertTransformers, Sentence for sentence-Transformer, BertSentence for Bert Sentence Transformers |
| `complexity_compute` | euclidean,cosine,wasserstein,distance,jensenshannon,bhattacharyya                                         |
| `embeddings_keep` | 1 keep emmbeding                                                                                          |

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
  title          = {Hugging-Face-based complexity models as {DUUI} component},
  year           = {2023},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-Complexity}
}

```
