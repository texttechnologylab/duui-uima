# Paragraph segmentation with chonky

DUUI component for semantic paragraph segmentation with [chonky](https://github.com/mirth/chonky) 0.1.7, model
[`mirth/chonky_mmbert_small_multilingual_1`](https://huggingface.co/mirth/chonky_mmbert_small_multilingual_1)
(revision `4a92de55`, downloaded when the image is built, the container runs offline; the build fails if the model's
`main` is no longer this revision, because chonky loads the tokenizer without a revision). The model is multilingual and
does not use the document language. chonky always runs on the whole text (its pipeline uses a stride of half the
1024-token window for longer texts); its chunks are consecutive slices of the text, so their offsets are exact.
Existing line breaks can be added as extra paragraph boundaries. Paragraphs are trimmed, whitespace-only segments dropped.

Runs on the GPU if one is available (`torch.cuda.is_available()`), else on the CPU, in full precision. The image uses
torch 2.14.1 with CUDA 13 (NVIDIA driver ≥ 580 needed for the GPU).

# How To Use

For using duui-chonky as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm --gpus all -p 1000:9714 docker.texttechnologylab.org/duui-chonky:latest
```

## Run within DUUI

```java
composer.add(
    new DUUIRemoteDriver.Component("http://localhost:1000")
        .withParameter("keep_line_breaks", "none")
);
```

### Parameters

| Name | Description | Default |
|---|---|---|
| `keep_line_breaks` | Existing line breaks added as boundaries to those of chonky: `none` (chonky only), `line` (every line break), `blank_line` (two line breaks with only other whitespace between them) | `none` |

### Input / output types

| | Types |
|---|---|
| Input | CAS text |
| Output | `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph`, `org.texttechnologylab.annotation.AnnotatorMetaData` (per paragraph), `org.texttechnologylab.annotation.DocumentModification` |

## Tests

Start the service (e.g. `docker run --rm -p 9805:9714 docker.texttechnologylab.org/duui-chonky:1.0.0`), then
`mvn test -Dchonky.url=http://localhost:9805`.
