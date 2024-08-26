[![Version](https://img.shields.io/static/v1?label=duui-transformers-toxic&message=0.2.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-toxic/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.8&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.22.1&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.1.1&color=red)]()

# Transformers Toxic

DUUI implementation for selected Hugging-Face-based transformer [Toxic tools](https://huggingface.co/models?sort=trending&search=toxic).
and for [Detoxify](https://github.com/unitaryai/detoxify) model.
## Included Models

| Name                                                                                                                                                 | Model-Name                                                   | Revision                                 | Languages                              |
|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------|------------------------------------------|----------------------------------------|
| [Detoxify](https://github.com/unitaryai/detoxify)                                                                                                    | Detoxify                                                                   | 773203c10bcf0e8d801b4be8c93cfd97ffe5c2e0 | EN, FR, ES, IT, PT, TR, RU             |
| [xlm_roberta_base_multilingual_toxicity_classifier_plus](https://huggingface.co/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus) | EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus        | 0126552291025f2fc854f5acdbe45b2212eabf4a | Multilingual                           |
| [one-for-all-toxicity-v3](https://huggingface.co/FredZhang7/one-for-all-toxicity-v3)                                                                 | FredZhang7/one-for-all-toxicity-v3                                         | a2996bd4495269071eaf5daf73512234c33cb3d2 | Multilingual                           |
| [CitizenLabDotCo toxic model](https://huggingface.co/citizenlab/distilbert-base-multilingual-cased-toxicity)                                         | citizenlab/distilbert-base-multilingual-cased-toxicity                     | b4532a8b095d1886a7b5dff818331ecc88a855ae | EN, FR, NL, PT, IT, SP, DE, PL, DA, AF |
| [toxic-comment-model](https://huggingface.co/martin-ha/toxic-comment-model)                                                                          | martin-ha/toxic-comment-model                                              | 9842c08b35a4687e7b211187d676986c8c96256d | EN                                     |
| [german_toxicity_classifier_plus_v2](https://huggingface.co/EIStakovskii/german_toxicity_classifier_plus_v2)                                         | EIStakovskii/german_toxicity_classifier_plus_v2                            | 1bcb7d11ffc9267111c7be1dad0d7ca2fbf73928          | EN                                     |
| [Aira-ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel)                                                                             | nicholasKluge/ToxicityModel                                                | d40cd71847981a0868aa3554c96c0aaf8c189753 | EN                                     |
 | [Overview of the Multilingual Text Detoxification Task at PAN 2024](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier)                 | textdetox/xlmr-large-toxicity-classifier | 4e7bb4af12318b16a6a04b14ca3420e47bad1586 | Multilingual                           |
| [ToxiGen](https://huggingface.co/tomh/toxigen_roberta)                                                                                                                                          | tomh/toxigen_roberta | 0e65216a558feba4bb167d47e49f9a9e229de6ab | EN                                     |
# How To Use

For using duui-transformers-toxic as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-transformers-toxic:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-toxic/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-toxic:latest")
        .withParameter("model_name", model)
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
  title          = {Hugging-Face-based toxic models  and detoxify models as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-toxic}
}

```
