[//]: # ([![Version]&#40;https://img.shields.io/static/v1?label=duui-transformers-emotion&message=0.2.0&color=blue&#41;]&#40;https://docker.texttechnologylab.org/v2/duui-transformers-emotion/tags/list&#41;)

[//]: # ([![Version]&#40;https://img.shields.io/static/v1?label=Python&message=3.10&color=green&#41;]&#40;&#41;)

[//]: # ([![Version]&#40;https://img.shields.io/static/v1?label=Transformers&message=4.41.2&color=yellow&#41;]&#40;&#41;)

[//]: # ([![Version]&#40;https://img.shields.io/static/v1?label=Torch&message=2.3.0&color=red&#41;]&#40;&#41;)

# Image To Text

DUUI implementation for selected Hugging-Face-based [Image-to-Text](https://huggingface.co/models?pipeline_tag=image-to-text) models,

| Name                                                  | Revision                                 | Task      |
|-------------------------------------------------------|------------------------------------------|-----------|
| microsoft/kosmos-2-patch14-224   | e91cfbcb4ce051b6a55bfb5f96165a3bbf5eb82c | Grounding |

# How To Use

For using duui-image-to-text as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

[//]: # (```)

[//]: # (docker run -p 9714:9714 docker.texttechnologylab.org/duui-transformers-emotion:latest)

[//]: # (```)

Find all available image tags here: https://docker.texttechnologylab.org/v2/duui-transformers-emotion/tags/list

## Run within DUUI

```Java

composer.add(

    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-image-to-text:latest")

        .withParameter("model_name", "MilaNLProc/xlm-emo-t")

        .withParameter("prompt", "<grounding>An image of")

);
```

### Parameters

| Name         | Description                        |
|--------------|------------------------------------|
| `model_name` | Model to use, see table above      |
| `prompt`     | prompts for the model, default is `<grounding>An image of` |


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
  author         = {Abusaleh, Ali},
  title          = {Image-to-Text as {DUUI} component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-image-to-text}
}

```
