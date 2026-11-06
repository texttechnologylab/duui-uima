# Tokenization with SoMaJo

DUUI component for tokenization with [SoMaJo](https://github.com/tsproisl/SoMaJo) 2.5.0 (Proisl & Uhrig 2016). Every
existing `Sentence` is tokenized separately (SoMaJo's own sentence splitting is off), so no token crosses a sentence
boundary; without `Sentence` annotations the whole text is tokenized as one unit. Writes plain DKPro `Token`s with
SoMaJo's character offsets in the original text (a token whose original spelling contains whitespace, e.g. `- >`
normalized to `->`, keeps its original span).

| Document language | SoMaJo model |
|---|---|
| `de` | `de_CMC` |
| `en` | `en_PTB` |

Other languages are an error. Camel case is not split (`split_camel_case=False`, the SoMaJo default).

# How To Use

For using duui-somajo as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-somajo:latest
```

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-somajo:latest")
);
```

### Parameters

None.

### Input / output types

| | Types |
|---|---|
| Input | CAS text and language, `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence` (optional) |
| Output | `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token`, `org.texttechnologylab.annotation.AnnotatorMetaData` (per token), `org.texttechnologylab.annotation.DocumentModification` |

## Tests

Start the service (e.g. `docker run --rm -p 9803:9714 docker.texttechnologylab.org/duui-somajo:1.0.0`), then
`mvn test -Dsomajo.url=http://localhost:9803`.

# Cite

Thomas Proisl and Peter Uhrig. 2016. SoMaJo: State-of-the-art tokenization for German web and social media texts. In
Proceedings of the 10th Web as Corpus Workshop (WAC), pages 57–62, Berlin. Association
for Computational Linguistics. https://aclanthology.org/W16-2607
