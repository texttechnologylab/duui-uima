
[![Version](https://img.shields.io/static/v1?label=duui-multimodal\&message=0.1.0\&color=blue)](https://docker.texttechnologylab.org/v2/duui-multimodal/tags/list)
[![Python](https://img.shields.io/static/v1?label=Python\&message=3.12\&color=green)]()
[![Transformers](https://img.shields.io/static/v1?label=Transformers\&message=4.48.2\&color=yellow)]()
[![Torch](https://img.shields.io/static/v1?label=Torch\&message=2.6.0\&color=red)]()

# DUUI Multimodal Component

DUUI implementation for **multimodal Hugging Face models** that support combinations of:

* Text
* Image
* Audio
* Video (via uniform frame sampling and audio extraction using `ffmpeg`)

Supported models include variants like `microsoft/Phi-4-multimodal-instruct`.

---

## Supported Modes

| Mode               | Description                                                         |
|--------------------|---------------------------------------------------------------------|
| `text_only`        | Process raw text prompts                                            |
| `image_only`       | Process images and prompt combinations                              |
| `frames_only`      | Process sequences of image frames with a shared prompt              |
| `audio_only`       | Process audio files with accompanying text prompts                  |
| `video_only`       | Process video input: extracts frames (every 5th), audio, and prompt |
| `frames_and_audio` | process **separate** frames and audio (provide them explicitly)     |


---

## How To Use

Requires the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Start Docker Container

```bash
docker run -p 9714:9714 docker.texttechnologylab.org/duui-multimodal:latest
```

Find available image tags: [Docker Registry](https://docker.texttechnologylab.org/v2/duui-multimodal/tags/list)

---

## Use within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-multimodal:latest")
        .withParameter("model_name", "microsoft/Phi-4-multimodal-instruct")
        .withParameter("mode", "video")  // Can be: text_only, image_only, audio, frames_only, video
);
```

---

## Parameters

| Name         | Description                                    |
| ------------ | ---------------------------------------------- |
| `model_name` | Name of the multimodal model to use            |
| `mode`       | Processing mode: text\_only, image\_only, etc. |
| `prompt`     | Prompt passed alongside media inputs           |

---

## Cite

If you want to use the DUUI image, please cite the following:

**Leonhardt et al. (2023)**
*"Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI."*
Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399.
\[[LINK](https://aclanthology.org/2023.findings-emnlp.29)] \[[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

---

## BibTeX

```bibtex
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399},
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf}
}

@misc{abusaleh:2025,
  author         = {Abusaleh, Ali},
  title          = {Multimodal Inference as {DUUI} Component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-multimodal}
}


