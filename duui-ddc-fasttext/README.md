[![Version](https://img.shields.io/static/v1?label=duui-transformers-sentiment&message=2.3.3&color=blue)](https://docker.texttechnologylab.org/v2/textimager-duui-ddc-fasttext/tags/list)

# text2ddc (DUUI DDC fastText)

DUUI implementation for [text2ddc](https://www.texttechnologylab.org/applications/text2ddc/).

## Included Models

| Name        | Languages |
| ----------- | --------- |
| ddc1_dim100 | EN, DE    |
| ddc1_dim300 | EN, DE    |
| ddc1_dim100_ml | Multi  |
| ddc1_dim300_ml | Multi  |
| ddc2_dim100 | EN, DE    |
| ddc2_dim300 | EN, DE    |
| ddc2_dim100_ml | Multi  |
| ddc2_dim300_ml | Multi  |
| ddc3_dim100 | EN, DE    |
| ddc3_dim300 | EN, DE    |
| ddc3_dim100_ml | Multi  |
| ddc3_dim300_ml | Multi  |

# How To Use

For using text2ddc as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/textimager-duui-ddc-fasttext:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/textimager-duui-ddc-fasttext/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/textimager-duui-ddc-fasttext:latest")
        .withScale(iWorkers)
        .withImageFetching()
);
```

### Parameters

| Name | Description |
| ---- | ----------- |
| `ddc_variant` | DDC variant to use, see model table above |
| `selection`   | Use `text` to process the full document text or any selectable UIMA type class name |

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
  title          = {text2ddc as DUUI component},
  year           = {2022},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-ddc-fasttext}
}

@inproceedings{Uslu:Mehler:Baumartz:2019,
  author = "Uslu, Tolga and Mehler, Alexander and Baumartz, Daniel",
  booktitle = "{Proceedings of the 20th International Conference on Computational Linguistics and Intelligent Text Processing, (CICLing 2019)}",
  location = "La Rochelle, France",
  series = "{CICLing 2019}",
  title = "{Computing Classifier-based Embeddings with the Help of text2ddc}",
  year = 2019
}
```

