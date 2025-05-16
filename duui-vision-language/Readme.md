[![Version](https://img.shields.io/static/v1?label=duui-molmo&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-molmo/tags/list)
[![Python](https://img.shields.io/static/v1?label=Python&message=3.10&color=green)]()
[![Transformers](https://img.shields.io/static/v1?label=Transformers&message=4.38.2&color=yellow)]()
[![Torch](https://img.shields.io/static/v1?label=Torch&message=2.2.0&color=red)]()

# DUUI Vision Language

**DUUI integration for Vision-Language Inference**, powered by **AllenAI's Molmo models**, served via HuggingFace or **vLLM**.

---

##  Supported Models

| Model Class            | HuggingFace Model ID              | Backend |
|------------------------|-----------------------------------|---------|
| `MolmoE1BModel`        | `allenai/MolmoE-1B-0924`          | HF      |
| `Molmo7BOModel`        | `allenai/Molmo-7B-O-0924`         | HF      |
| `Molmo7BDModel`        | `allenai/Molmo-7B-D-0924`         | HF      |
| `Molmo72BModel`        | `allenai/Molmo-72B-0924`          | HF      |
| `Molmo7BDModelVLLM`    | `allenai/Molmo-7B-D-0924`         | vLLM    |

The `Molmo7BDModelVLLM` is optimized for vLLM serving and uses OpenAI-compatible API calls.

---

##  Supported Modes

| Mode         | Description                        |
|--------------|------------------------------------|
| `text_only`  | Standard chat / text completion    |
| `image_only` | Image+text multimodal inference    |

---

##  Quick Start

Requires the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Start via Docker

```bash
docker run -p 9714:9714 docker.texttechnologylab.org/duui-molmo:latest
````

---

## 🧪 Example

### Input Prompt

```text
"Describe this image."
```

### Input Image

```
https://picsum.photos/id/237/536/354
```

### Output (Example Response)

```text
"This is a black dog sitting on a wooden deck, looking directly at the camera with a curious expression."
```

---

##  DUUI Integration

Example DUUI driver config:

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-vision-language:latest")
        .withParameter("model_name", "Molmo7BDModelVLLM")
        .withParameter("mode", "text_only")
);
```

---

## Parameters

| Name         | Description                                           |
| ------------ | ----------------------------------------------------- |
| `model_name` | One of the supported Molmo model classes              |
| `mode`       | `text_only` or `image_only`                           |
| `prompt`     | Optional user prompt (used with image or text inputs) |

---

## Advanced Run with GPU (Podman)

```bash
podman run -d \
  --device nvidia.com/gpu=all \
  -p 9991:9714 \
  -p 6650:6650 \
  docker.texttechnologylab.org/duui-vision-language:latest
```

---

## 📚 BibTeX

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
