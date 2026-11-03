# Universal Dependencies with Stanza

DUUI component for POS tags, lemmas, morphological features and dependencies in
[Universal Dependencies](https://universaldependencies.org/) with [Stanza](https://stanfordnlp.github.io/stanza/) 1.15.0
(Qi et al. 2020), package `default_accurate`, on given sentences and tokens.

| Document language | Stanza package | Transformer (pos, depparse) |
|---|---|---|
| `de` | `de/default_accurate` (UD German, combined) | `german-nlp-electra` |
| `en` | `en/default_accurate` (UD English, combined) | `electra-large` |

Other languages are an error. The processing follows `StanzaLocal` in `duui-coh-metrix/docs/preprocess.py`, the reference
preprocessing of Coh-Metrix 0.4.0-py:

- **Input:** the plain DKPro `Sentence` and `Token` annotations of an earlier tool, e.g. duui-wtpsplit and duui-somajo.
  Only exactly these types count; subtypes such as `SpacyToken` or the output of an earlier run don't. Stanza runs on these
  tokens (`tokenize_pretokenized`), with at most 500 sentences per call and `depparse_batch_size` 1000.
- **German contractions:** Stanza's multi-word token step doesn't run on given tokens, so German preposition + article
  contractions are split with a table (`DE_CONTRACTIONS`): "zum" → "zu" + "dem".
  - The table contains the expansions of UD German GSD/HDT plus hinterm, hinters, überm, vors (STTS).
  - Only all-lowercase or capitalised tokens are split ("Im" → "In dem", but not "IM").
  - "am" before a superlative stays one word (STTS PTKA, as in HDT: "am besten").
  - Stanza's own `resplit_mwt` was worse on UD German GSD/HDT: 80 wrong tokens against 9
    (`duui-coh-metrix/docs/stanza-contraction-split/REPORT.md`).
- **English** tokens are used as given (e.g. SoMaJo `en_PTB` already splits "do n't" as UD English does).
- **Long sentences:** a sentence of more than 500 tokens is parsed in pieces of at most 500 tokens. A piece the transformer
  can't take (more than its 512 subword positions) is halved until it fits. Each piece becomes a `StanzaSentence` of its
  own, and a warning is logged.

## Output

- **`org.texttechnologylab.uima.type.stanza.StanzaSentence`:** a complete sentence layer of its own. The given sentences
  stay unchanged; remove them with duui-remove-annotations if only the Stanza layer should remain.
- **`org.texttechnologylab.uima.type.stanza.StanzaToken`:** one per word. Both types are subtypes of the DKPro types,
  defined in the TTLab UIMATypeSystem. The two words of a split contraction share the span of their token, and each has
  its word as `Token.form` (`TokenForm`).
- **`POS`:** `posValue` is the language-specific tag (STTS, Penn Treebank), `coarseValue` is the UPOS tag.
- **`Lemma`.**
- **`MorphologicalFeatures`:** `value` is the complete UD feature string, including e.g. `Number[psor]`. The individual
  features (gender, number, case, …, `Polarity` as negative) are also set where DKPro has a field for them.
- **`Dependency` (UD labels, flavor `basic`) and `ROOT`:** the root has dependency type `--` and itself as governor, as in
  duui-spacy. `Token.parent` is set to the governor.
- **`AnnotatorMetaData`** per annotation, and one **`DocumentModification`**.

The models are downloaded when the image is built; the container runs offline. It uses the GPU if one is available.

# How To Use

For using duui-stanza as a DUUI image it is necessary to use the [Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

## Start Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/duui-stanza:latest
```

## Run within DUUI

```java
composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-stanza:latest")
);
```

### Parameters

None.

### Input / output types

| | Types |
|---|---|
| Input | CAS text and language, `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence`, `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token` (exactly these types) |
| Output | `org.texttechnologylab.uima.type.stanza.StanzaSentence`, `org.texttechnologylab.uima.type.stanza.StanzaToken`, `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.TokenForm`, `de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS`, `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma`, `de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures`, `de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency`, `de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT`, `org.texttechnologylab.annotation.AnnotatorMetaData`, `org.texttechnologylab.annotation.DocumentModification` |

## Tests

Start the service (e.g. `docker run --rm -p 9806:9714 docker.texttechnologylab.org/duui-stanza:1.0.0`), then
`mvn test -Dstanza.url=http://localhost:9806`.

# Cite

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton and Christopher D. Manning. 2020. Stanza: A Python Natural Language
Processing Toolkit for Many Human Languages. In Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics: System Demonstrations, pages 101–108. https://aclanthology.org/2020.acl-demos.14
