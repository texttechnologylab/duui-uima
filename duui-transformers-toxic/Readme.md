[![Version](https://img.shields.io/static/v1?label=duui-transformers-toxic&message=0.4.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-transformers-toxic-one_for_all_toxicity_v3/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=4.22.1&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.5.1&color=red)]()

# Transformers Toxic

DUUI implementation for selected Hugging-Face-based transformer [Toxic tools](https://huggingface.co/models?sort=trending&search=toxic).
and for [Detoxify](https://github.com/unitaryai/detoxify) model.
## Included Models
| Name-DUUI                                  |                                                                                                                                                     | Name                                                               | Model-Name                             | Revision                                 | Languages                              |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|----------------------------------------|------------------------------------------|----------------------------------------|
| multi-toxic-classifier-plus      | [xlm_roberta_base_multilingual_toxicity_classifier_plus](https://huggingface.co/EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus) | EIStakovskii/xlm_roberta_base_multilingual_toxicity_classifier_plus | 0126552291025f2fc854f5acdbe45b2212eabf4a | EN,RU,FR,DE                             |
| one_for_all_toxicity_v3                    | [one-for-all-toxicity-v3](https://huggingface.co/FredZhang7/one-for-all-toxicity-v3)                                                                | FredZhang7/one-for-all-toxicity-v3                                 | a2996bd4495269071eaf5daf73512234c33cb3d2 | Multilingual                             |
| distilbert_base_multilingual_cased_toxicity | [CitizenLabDotCo toxic model](https://huggingface.co/citizenlab/distilbert-base-multilingual-cased-toxicity)                                        | citizenlab/distilbert-base-multilingual-cased-toxicity             | b4532a8b095d1886a7b5dff818331ecc88a855ae | EN, FR, NL, PT, IT, SP, DE, PL, DA, AF   |
| toxic_comment_model                        | [toxic-comment-model](https://huggingface.co/martin-ha/toxic-comment-model)                                                                         | martin-ha/toxic-comment-model                                      | 9842c08b35a4687e7b211187d676986c8c96256d | EN                                       |
| german_toxicity_classifier_plus_v2         | [german_toxicity_classifier_plus_v2](https://huggingface.co/EIStakovskii/german_toxicity_classifier_plus_v2)                                        | EIStakovskii/german_toxicity_classifier_plus_v2                    | 1bcb7d11ffc9267111c7be1dad0d7ca2fbf73928 | DE                                       |
| aira_toxicity_model                        | [Aira-ToxicityModel](https://huggingface.co/nicholasKluge/ToxicityModel)                                                                            | nicholasKluge/ToxicityModel                                        | 900a6eab23ddd93f6c282f1752eb1fb5e9879d86 | EN                                       |
| xlmr_large_toxicity_classifier             | [Overview of the Multilingual Text Detoxification Task at PAN 2024](https://huggingface.co/textdetox/xlmr-large-toxicity-classifier)                | textdetox/xlmr-large-toxicity-classifier                           | b9c7c563427c591fc318d91eb592381ae2fbde66 | Multilingual                             |
| toxigen                                    | [ToxiGen](https://huggingface.co/tomh/toxigen_roberta)                                                                                              | tomh/toxigen_roberta                                               | 0e65216a558feba4bb167d47e49f9a9e229de6ab | EN                                       |
| detoxify                                   | [Detoxify](https://github.com/unitaryai/detoxify)                                                                                                   | Detoxify                                                           | 8f56f302bf8cf2673c2132fc2c2f5b2ca804815f | EN, FR, ES, IT, PT, TR, RU               |
| roberta_toxicity_classifier                | [roberta_toxicity_classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier)                                                             | s-nlp/roberta_toxicity_classifier                                  | 048c25bb1e199b98802784f96325f4840f22145d | EN                                       |
| para-detox                                 | [para-detox](https://huggingface.co/garak-llm/roberta_toxicity_classifier)                                                                          | garak-llm/roberta_toxicity_classifier                              | fb7e9d615fc8c59d2e70466a831ed70d6f6f895a | EN                                       |
| russe-2022                                 | [russe-2022](https://huggingface.co/s-nlp/russian_toxicity_classifier)                                                                              | s-nlp/russian_toxicity_classifier                                  | 0694e1f99efc08e73479e5c6f06c7bbe393aca89 | RU                                       |
| xlm-multi-toxic                            | [xlm-multi-toxic](https://huggingface.co/malexandersalazar/xlm-roberta-large-binary-cls-toxicity)                                                   | malexandersalazar/xlm-roberta-large-binary-cls-toxicity            | 6968ce7aa290a1bb2bbada047a3491aa048e2bd3 | EN,DE,FR,IT,PT,TH,HI,ES                  |
| rubert-toxic                               | [rubert-toxic](https://huggingface.co/sismetanin/rubert-toxic-pikabu-2ch)                                                                           | sismetanin/rubert-toxic-pikabu-2ch                                 | 1e5d55aeca25ab0a91725abc08821694de7dd5ea | RU                                       |
| textdetox-glot500                          | [textdetox-glot500](https://huggingface.co/textdetox/glot500-toxicity-classifier)                                                                   | textdetox/glot500-toxicity-classifier                              | 4c2e8b298c4c7980d23566e92ab68b53f30db025 | EN,FR,IT,ES,RU,UK,AR,HI,JA,ZH,DE,TT,HE,AM |
| textdetox-bert                             | [textdetox-bert](https://huggingface.co/textdetox/bert-multilingual-toxicity-classifier)                                                            | textdetox/bert-multilingual-toxicity-classifier                                 | 0667d0fbb85a1ea7b1e3a1f2a9a2901f5ce8c16c | EN,FR,IT,ES,RU,UK,AR,HI,JA,ZH,DE,TT,HE,AM |
| toxicity-classifier-uk                     | [toxicity-classifier-uk](https://huggingface.co/dardem/xlm-roberta-large-uk-toxicity)                                                               | toxicity-classifier-uk                                  | 6e2c8c305cc7ccff14a6dfe3d8fdd83d6556f514 | UK                                       |
| toxdect                                    | [toxdect](https://huggingface.co/Xuhui/ToxDect-roberta-large)                                                                                | Xuhui/ToxDect-roberta-large                                 | 7b97c89938cb241d3ae9235257bbe4916d4f0c75 | EN                                       |

# How To Use

For using duui-transformers-toxic as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-transformers-toxic-[Name-DUUI]:latest
```

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-toxic-[Name-DUUI]/tags/list

## Run within DUUI

```
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-transformers-toxic-[Name-DUUI]:latest")
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
  title          = {Hugging-Face-based toxic models  and detoxify models as {DUUI} component},
  year           = {2024},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-transformers-toxic}
}

```
