[![Version](https://img.shields.io/static/v1?label=duui-ner&message=0.1.0&color=blue)](https://docker.texttechnologylab.org/v2/duui-ner/tags/list)
[![Version](https://img.shields.io/static/v1?label=Python&message=3.12&color=green)]()
[![Version](https://img.shields.io/static/v1?label=Transformers&message=5.1.0&color=yellow)]()
[![Version](https://img.shields.io/static/v1?label=Torch&message=2.11.0&color=red)]()
[![Version](https://img.shields.io/static/v1?label=GLiNER&message=0.2.26&color=orange)]()
[![Version](https://img.shields.io/static/v1?label=GLiNER2&message=1.3.1&color=orange)]()

# Transformers NER

DUUI implementation for selected transformer-based Named Entity Recognition (NER) models. The component is designed for use with the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

The component supports one NER model per Docker image/container. Each image is built with a single `MODEL_NAME` and exposes the DUUI endpoints for type system, Lua communication layer, documentation, and processing.

## Included Models

| Image suffix / `MODEL_SPECNAME` | `MODEL_NAME` | Model source | Model version | Languages | Backend |
| --- | --- | --- | --- | --- | --- |
| `gliner-multi-v2-1` | `gliner` | https://huggingface.co/urchade/gliner_multi-v2.1 | `443d26d654e0324125a96bebd8e796c14ff2efe6` | Multilingual | GLiNER |
| `gliner2-multi-v1` | `gliner2` | https://huggingface.co/fastino/gliner2-multi-v1 | `cc151f5b0ce4f7010c3ae8884527dd43dddf9d21` | Multilingual | GLiNER2 |
| `roberta-ner-multilingual` | `roberta-ner-multilingual` | https://huggingface.co/julian-schelb/roberta-ner-multilingual | `d0a19147f3bb0065c8091459e3d35405ce9d48da` | Multilingual | HuggingFace token-classification |
| `wikineural-multilingual-ner` | `wikineural-multilingual-ner` | https://huggingface.co/Babelscape/wikineural-multilingual-ner | `bed6ee7a45d2827b6c90a4fd7983f0241ae0a5c1` | Multilingual | HuggingFace token-classification |
| `xlm-r-ner-40-lang` | `xlm-r-ner-40-lang` | https://huggingface.co/nbroad/jplu-xlm-r-ner-40-lang | `7f7f0fe9bc946a9848611aff079f556387687216` | Multilingual / 40 languages | HuggingFace token-classification |

## Annotation Types

The component creates UIMA NER annotations from the model output. Standard NER labels are mapped to DKPro NER types where possible, for example:

| Label | UIMA type |
| --- | --- |
| `PER`, `person` | `de.tudarmstadt.ukp.dkpro.core.api.ner.type.Person` |
| `ORG`, `organization` | `de.tudarmstadt.ukp.dkpro.core.api.ner.type.Organization` |
| `LOC`, `location` | `de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location` |
| `taxon`, `taxa` | `org.texttechnologylab.annotation.type.Taxon` |
| other labels | `de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity` |

The `taxon` label is mapped to the TTLab taxon type:

```text
org.texttechnologylab.annotation.type.Taxon
```

The delivered type system must include this type if taxon annotations should be created as `Taxon` instead of falling back to a generic `NamedEntity`.

## Requirements

The container uses Python 3.12 and the following core Python dependencies:

| Package | Version |
| --- | --- |
| `gliner` | `0.2.26` |
| `gliner2[local]` | `1.3.1` |
| `transformers` | `5.1.0` |
| `torch` | `2.11.0` |
| `fastapi` | `0.110.0` |
| `dkpro-cassis` | `0.9.1` |
| `uvicorn[standard]` | `0.27.1` |
| `pydantic-settings` | `2.0.2` |

See `requirements.txt` for the full dependency list.

# How To Use

## Start Docker container

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-ner-[modelname]:latest
```

Example:

```bash
docker run --rm -p 9714:9714 docker.texttechnologylab.org/duui-ner-wikineural-multilingual-ner:latest
```

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-ner-[modelname]:latest")
        .withParameter(
            "selection",
            "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        )
);
```

With optional runtime parameters:

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-ner-[modelname]:latest")
        .withParameter(
            "selection",
            "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
        )
        .withParameter("threshold", "0.5")
        .withParameter("batch_size", "8")
        .withParameter("labels", "person,organization,location,date,event,product,taxon,other")
);
```

### Parameters

| Name | Default | Description |
| --- | --- | --- |
| `selection` | required | Use `text` to process the full document text or any selectable UIMA type class name, e.g. `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence`. |
| `threshold` | `0.5` | Confidence threshold for GLiNER/GLiNER2. HuggingFace token-classification models may ignore this value. |
| `batch_size` | `8` | Batch size used during prediction. |
| `labels` | `person,organization,location,date,event,product,taxon,other` | Candidate labels for GLiNER/GLiNER2. HuggingFace token-classification models use their trained label set. |

## Runtime behavior

- Each Docker image/container uses exactly one model.
- `MODEL_NAME` selects the backend model alias used by the Python service.
- `MODEL_VERSION` is used as model metadata in the DUUI response.
- `MODEL_SOURCE` and `MODEL_LANG` are also returned as metadata.
- Runtime parameters such as `threshold`, `batch_size`, and `labels` are passed via DUUI `.withParameter(...)`.

# Cite

If you use this DUUI image, please cite DUUI as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler. (2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI." Findings of the Association for Computational Linguistics: EMNLP 2023, 385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)] [[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

## BibTeX

```bibtex
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
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf}
}

@misc{Bagci:2026,
  author       = {Bagci, Mevlüt},
  title        = {Transformer-based Named Entity Recognition models as {DUUI} component},
  year         = {2026},
  howpublished = {https://github.com/texttechnologylab/duui-uima}
}
```