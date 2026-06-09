# DUUI Anonymize

PII detection and redaction for [TTLab DUUI](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface) using transformer-based token classification models.

The service runs as a FastAPI server, receives plain text from DUUI via a Lua communication script, detects personally identifiable information (PII) spans, and returns structured annotations written as `Anomaly` types into the UIMA CAS.

---

## Supported Models

| Model | Labels | Notes |
|---|---|---|
| `openai/privacy-filter` **(default)** | 8 broad categories | No special options needed |
| `OpenMed/privacy-filter-nemotron` | 55 fine-grained categories | Requires `trust_remote_code=true` |

### `openai/privacy-filter` — output labels

`private_person`, `private_email`, `private_phone`, `private_address`, `private_url`, `private_date`, `account_number`, `secret`

### `OpenMed/privacy-filter-nemotron` — output labels (selected)

`first_name`, `last_name`, `email`, `phone_number`, `ssn`, `tax_id`, `street_address`, `city`, `postcode`, `credit_debit_card`, `account_number`, `password`, `ipv4`, `ipv6`, `mac_address`, `api_key`, `medical_record_number`, `coordinate`, … (55 total)

Use the OpenMed model when you need fine-grained entity types (e.g. distinguishing `first_name` from `last_name`, or `ssn` from a generic person span).

---

## Input / Output

**Input:** plain text in the CAS Sofa. Optional selection offsets narrow the window that is analysed.

**Output:** `de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly` annotations on the default CAS view, each with:
- `begin` / `end` — character offsets into the original text
- `category` — entity label (e.g. `private_person`, `first_name`)
- `description` — replacement text used (the placeholder string, or empty in remove mode)

The redacted text is written to a secondary CAS view named `opf_redacted`.

---

## Options (passed via DUUI parameters)

| Option | Type | Default | Description |
|---|---|---|---|
| `model` | string | `openai/privacy-filter` | HuggingFace model ID |
| `trust_remote_code` | bool | `false` | Required for `OpenMed/privacy-filter-nemotron` |
| `device` | string | auto | `cpu` or `cuda` |
| `mode` | string | `placeholder` | See modes below |
| `placeholder` | string | `[{label}]` | Custom replacement text (placeholder mode only) |
| `selection_begin` | int | — | Start offset of the text window to analyse |
| `selection_end` | int | — | End offset of the text window to analyse |

---

## Modes

| Mode | Effect on `redacted_text` |
|---|---|
| `placeholder` **(default)** | Detected spans replaced with `[label]`, e.g. `[private_person]` |
| `remove` | Detected spans deleted from the text |
| `pseudo` | Input returned unchanged (stub — not yet implemented) |

---

## Running the Service

```bash
cd src/main/python
uvicorn duui_anonymize:app --host 0.0.0.0 --port 9714 --workers 1
```

API docs available at `http://localhost:9714/api`.

---

## Docker

CPU image:
```bash
docker build -f src/main/docker/Dockerfile -t duui-anonymize:latest .
docker run -p 9714:9714 duui-anonymize:latest
```

GPU (CUDA 11.8) image:
```bash
docker build -f src/main/docker/Dockerfile-cuda -t duui-anonymize:cuda .
docker run --gpus all -p 9714:9714 duui-anonymize:cuda
```

---

## DUUI Usage Example

```java
composer.add(new DUUIRemoteDriver.Component("http://localhost:9714"));
```

With the OpenMed model:
```java
composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
        .withParameter("model", "OpenMed/privacy-filter-nemotron")
        .withParameter("trust_remote_code", "true")
        .withParameter("mode", "placeholder"));
```

---

## Project Structure

```
src/
  main/
    python/
      duui_anonymize.py     # FastAPI service + business logic
      communication.lua     # DUUI Lua serialization/deserialization
      typesystem.xml        # UIMA type system
    docker/
      Dockerfile            # CPU image (python:3.10)
      Dockerfile-cuda       # GPU image (cuda:11.8)
  test/
    java/
      AnonymizeTests.java   # JUnit 5 integration tests (service must be running)
    results/                # Per-test .json and .xmi output written here
requirements.txt
```

---

## Developer

**Ali Abusaleh**
Text Technology Lab (TTLab), Goethe University Frankfurt
[abusaleh@em.uni-frankfurt.de](mailto:abusaleh@em.uni-frankfurt.de) · [texttechnologylab.org](https://www.texttechnologylab.org)

```bibtex
@misc{abusaleh_duui_anonymize_2026,
  author       = {Abusaleh, Ali},
  title        = {{DUUI Anonymize}: PII Detection and Redaction Component for {DUUI}},
  year         = {2026},
  institution  = {Text Technology Lab, Goethe University Frankfurt},
  howpublished = {\url{https://github.com/texttechnologylab/duui-uima}},
  note         = {Part of the DockerUnifiedUIMAInterface (DUUI) component ecosystem}
}
```

---

## Citations

If you use this component, please also cite the underlying models.

### openai/privacy-filter (default model)

```bibtex
@misc{openai_privacy_filter_2025,
  author       = {OpenAI},
  title        = {{openai/privacy-filter}},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/openai/privacy-filter}}
}
```

Model card PDF: https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf

### OpenMed/privacy-filter-nemotron

```bibtex
@misc{openmed_privacy_filter_nemotron_2026,
  author       = {OpenMed},
  title        = {{OpenMed/privacy-filter-nemotron}: fine-grained PII extraction with 55 categories},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/OpenMed/privacy-filter-nemotron}}
}

@misc{openmed_2026,
  author       = {OpenMed},
  title        = {{OpenMed}: open models and resources for healthcare NLP},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/OpenMed}}
}

@misc{nemotron_pii_2025,
  author       = {NVIDIA},
  title        = {{Nemotron-PII}},
  year         = {2025},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/nvidia/Nemotron-PII}}
}
```
