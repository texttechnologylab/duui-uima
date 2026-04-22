[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)]()


# Suprisal

DUUI implementation for suprisal predictions for texts.

# How To Use

For using **duui-suprisal** as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 9715:9714 --gpus all docker.texttechnologylab.org/duui-suprisal:latest
```

Find all available image tags here: 
- https://docker.texttechnologylab.org/v2/duui-suprisal/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-suprisal:latest")
        .withParam("model", "goldfish-models/spa_latn_1000mb");
        .withParam("token_authentication", "myHuggingfaceToken");
        .withScale(iWorkers)
        .withImageFetching()
);
```

There is a sample version in the test folder.

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

@misc{Abrami:Bruera:2025,
  author         = {Abrami, Giuseppe and Bruera, Mercedes Martinez},
  title          = {Text Suprisal as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-suprisal}
}

```
**