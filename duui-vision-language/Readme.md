
[![Version](https://img.shields.io/static/v1?label=duui-molmo\&message=0.1.0\&color=blue)](https://docker.texttechnologylab.org/v2/duui-molmo/tags/list)
[![Python](https://img.shields.io/static/v1?label=Python\&message=3.10\&color=green)]()
[![Transformers](https://img.shields.io/static/v1?label=Transformers\&message=4.38.2\&color=yellow)]()
[![Torch](https://img.shields.io/static/v1?label=Torch\&message=2.2.0\&color=red)]()

# DUUI Vision Language

**DUUI integration for Vision language**, such as:

* AllenAI's multimodal Molmo models:
  * `allenai/Molmo-7B-O-0924`
  * `allenai/Molmo-7B-D-0924`
  * `allenai/MolmoE-1B-0924`
  * `allenai/Molmo-72B-0924`

Molmo supports vision-language inference using Hugging Face `transformers` and can be served efficiently using **vLLM**.

---

## Supported Input

| Mode         | Description                      |
| ------------ | -------------------------------- |
| `text_only`  | Pure text inference              |

[//]: # (| `image_only` | Provide an image and text prompt |)

---

## Quick Start

You need the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface) setup.

### Start Docker Container

```bash
docker run -p 9714:9714 docker.texttechnologylab.org/duui-molmo:latest
```

View tags here: [Docker Registry](https://docker.texttechnologylab.org/v2/duui-molmo/tags/list)

---

## Use with DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-vision-language:latest")
        .withParameter("model_name", "Molmo-7B-O-0924")
        .withParameter("mode", "image")
);
```

---

## Parameters

| Name         | Description                                  |
| ------------ | -------------------------------------------- |
| `model_name` | HF-compatible name (e.g. `allenai/Molmo...`) |
| `mode`       | `image_only` or `text_only`                  |
| `prompt`     | Optional text prompt for inference           |

---

## Example Output

Prompt:

> “Describe this image.”

Image:

> `https://picsum.photos/id/237/536/354`

Response:

> “This is a black dog sitting on a wooden deck, looking directly at the camera with a curious expression.”

---

## RUN the Image 

```bash
podman run -d   --device nvidia.com/gpu=all   -p 9991:9714   -p 6658:8000  docker.texttechnologylab.org/duui-vision-language:latest
```

## Citation

If you use the DUUI Molmo component, please cite:

**Leonhardt et al. (2023)**
*Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI.*
[Findings of EMNLP 2023, pp. 385–399](https://aclanthology.org/2023.findings-emnlp.29)
[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)

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
  title          = {Multimodal Inference with Molmo as a DUUI Component},
  year           = {2025},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-molmo}
}
```

