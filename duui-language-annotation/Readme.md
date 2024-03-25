[![Version](https://img.shields.io/static/v1?label=duui-language-annotation&message=0.2.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-language-annotation/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.38.2color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# Language Detection Tools

DUUI implementation for selected language detection Tools.
## Included Models

| Name | Revision                                  | URL                                                      |
|----|-------------------------------------------|----------------------------------------------------------|
| Google | a1b65d981fc40aad0763dd782acbc99ab40a6228  | https://github.com/shuyo/language-detection              |
| glcd3 | b48dc46512566f5a2d41118c8c1116c4f96dc661  | https://github.com/google/cld3                           |
| Spacy | 28266a0a15ef5180eb8540bd98ff1c7d14b74e1d  | https://github.com/davebulaval/spacy-language-detection  |
| Fasttext | 1142dc4c4ecbc19cc16eee5cdd28472e689267e6  | https://fasttext.cc/docs/en/language-identification.html |
| Glotlid | a9f8a6cf8af1668c09db74bbb427c6255c16bb03  | https://github.com/cisnlp/GlotLID                        |
| papluca/xlm-roberta-base-language-detection | 9865598389ca9d95637462f743f683b51d75b87b  |  https://huggingface.co/papluca/xlm-roberta-base-language-detection  |
| qanastek/51-languages-classifier | 966ca1a15a30f218ad48561943f046d809d4ed26 |  https://huggingface.co/qanastek/51-languages-classifier |
# How To Use

For using duui-language-annotation as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-language-annotation:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-language-annotation/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-language-annotation:latest")
        .withParameter("model_name", 'Google')
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
  title          = {Language detection tools as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-language-annotation}
}

```
