# Sentence segmentation with wtpsplit (Segment any Text)

DUUI component for sentence segmentation with [wtpsplit](https://github.com/segment-any-text/wtpsplit) 2.2.1, model
[SaT `sat-12l-sm`](https://huggingface.co/segment-any-text/sat-12l-sm) (revision `d70c72a9`, tokenizer
`FacebookAI/xlm-roberta-base` revision `e73636d4`; both downloaded when the image is built, the container runs offline).
SaT is multilingual and does not use the document language. Every existing `Paragraph` is split separately (one batch
per document), so no sentence crosses a paragraph boundary; without `Paragraph` annotations the whole text is split.
SaT's defaults are used (threshold chosen by SaT for the model). Sentences are trimmed, whitespace-only segments dropped.

Runs on the GPU if one is available (`torch.cuda.is_available()`), else on the CPU, in full precision. The image uses
torch 2.14.1 with CUDA 13 (NVIDIA driver ≥ 580 needed for the GPU).

# How To Use

For using duui-wtpsplit as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm --gpus all -p 1000:9714 docker.texttechnologylab.org/duui-wtpsplit:latest
```

## Run within DUUI

```java
composer.add(
    new DUUIRemoteDriver.Component("http://localhost:1000")
        .withParameter("split_on_input_newlines", "true")
);
```

### Parameters

| Name | Description | Default |
|---|---|---|
| `split_on_input_newlines` | `true`: every line break also ends a sentence (SaT default); `false`: line breaks are read as spaces, sentences may contain line breaks | `true` |

### Input / output types

| | Types |
|---|---|
| Input | CAS text, `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph` (optional) |
| Output | `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence`, `org.texttechnologylab.annotation.AnnotatorMetaData` (per sentence), `org.texttechnologylab.annotation.DocumentModification` |

## Tests

Start the service (e.g. `docker run --rm -p 9804:9714 docker.texttechnologylab.org/duui-wtpsplit:1.0.0`), then
`mvn test -Dwtpsplit.url=http://localhost:9804`.

# Cite

Markus Frohmann, Igor Sterner, Ivan Vulić, Benjamin Minixhofer, and Markus Schedl. 2024. Segment Any Text: A Universal
Approach for Robust, Efficient and Adaptable Sentence Segmentation. In Proceedings of the 2024 Conference on Empirical
Methods in Natural Language Processing, pages 11908–11941. https://aclanthology.org/2024.emnlp-main.665/
