# DUUI OCR

DUUI implementation for vision-language OCR models.

## Supported Models

| Name | Params | Languages | Supported Tasks |
| ---- | ------ | --------- | --------------- |
| [PaddlePaddle/PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | multilingual | ocr, table, formula, chart, spotting, seal |
| [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | multilingual | ocr, table, formula |

## Supported Tasks

| Task | PaddleOCR-VL Prompt | GLM-OCR Prompt | Description |
| ---- | ------------------- | -------------- | ----------- |
| `ocr` | `OCR:` | `Text Recognition:` | General text recognition |
| `table` | `Table Recognition:` | `Table Recognition:` | Table structure recognition |
| `formula` | `Formula Recognition:` | `Formula Recognition:` | LaTeX formula recognition |
| `chart` | `Chart Recognition:` | — | Chart content recognition |
| `spotting` | `Spotting:` | — | Text spotting with location |
| `seal` | `Seal Recognition:` | — | Seal text recognition |

## How To Use

Requires
[Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Run within DUUI

```java
// PaddleOCR-VL
composer.add(
    new DUUIDockerDriver.Component(
            "docker.texttechnologylab.org/duui-ocr:latest"
        )
        .withParameter("model_name",
            "PaddlePaddle/PaddleOCR-VL-1.5")
        .withParameter("task", "ocr")
);

// GLM-OCR
composer.add(
    new DUUIDockerDriver.Component(
            "docker.texttechnologylab.org/duui-ocr:latest"
        )
        .withParameter("model_name", "zai-org/GLM-OCR")
        .withParameter("task", "ocr")
);
```

### Parameters

| Name | Description | Default |
| ---- | ----------- | ------- |
| `model_name` | Model to use (see table above) | — |
| `task` | OCR task type | `ocr` |
| `max_new_tokens` | Maximum tokens to generate | `1024` |

### Input / Output

- **Input**: `org.texttechnologylab.annotation.type.Image`
  annotations in CAS (src can be base64 or file path)
- **Output**: `org.texttechnologylab.annotation.AnnotationComment`
  with key = task name, value = recognized text

## Cite

```bibtex
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data
               {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe
               and Baumartz, Daniel and Mehler, Alexander},
  booktitle = {Findings of the Association for Computational
               Linguistics: EMNLP 2023},
  year      = {2023},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399},
}

@misc{cui2026paddleocrvl15multitask09bvlm,
  title   = {PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM
             for Robust In-the-Wild Document Parsing},
  author  = {Cheng Cui and Ting Sun and Suyin Liang and others},
  year    = {2026},
  eprint  = {2601.21957},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
}

@misc{glmocr2026,
  title   = {GLM-OCR: A Multimodal OCR Model for Complex
             Document Understanding},
  author  = {Z.ai Team},
  year    = {2026},
  url     = {https://huggingface.co/zai-org/GLM-OCR},
}
```