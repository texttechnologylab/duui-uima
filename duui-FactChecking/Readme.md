[![Version](https://img.shields.io/static/v1?label=duui-factchecking-unieval&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/textimager-duui-factchecking-unieval/tags/list)
[![Version](https://img.shields.io/static/v1?label=duui-factchecking-nubia&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/textimager-duui-factchecking-nubia/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.8&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.34.1&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Transformers FactChecking

DUUI implementation for selected FactChecking Tools: [UniEval](https://github.com/maszhongming/UniEval) and [NUBIA](https://github.com/wl-research/nubia).
## Included Models

| Name                                                       | Revision                                 | Languages                              |
|------------------------------------------------------------|------------------------------------------|----------------------------------------|
| UniEval                                                    | d33e7b6cfebe97b2bafe435adbd818230d5a416a | EN                                     |
| NUBIA                                                      | ba6569605671e88217a14b2b218ce6974be73775 | EN                                     |
# How To Use

For using duui-FactChecking as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-factchecking-[modelname]:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-factchecking-unieval/tags/list
and 
https://docker.texttechnologylab.org/v2/duui-factchecking-nubia/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-factchecking-[modelname]:latest")
        .withScale(iWorkers)
        .withImageFetching()
);
```

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
  title          = {Fact checking tools as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-FactChecking}
}

```
