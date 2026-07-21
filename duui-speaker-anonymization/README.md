# Speaker Anonymization DUUI

DUUI component for multilingual speaker anonymization based on the
[IMS speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization/)
pipeline. It transcribes an audio document with Whisper large-v3, replaces the
speaker embedding with a GAN-generated artificial embedding, and synthesizes an
anonymized voice with IMS-Toucan.

Supported languages: English (`en`), German (`de`), French (`fr`), Italian
(`it`), Spanish (`es`), Portuguese (`pt`), Dutch (`nl`), Polish (`pl`), and
Russian (`ru`).

# How To Use

Using this component requires the
[Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).
A CUDA-capable GPU is recommended, but CPU processing is supported.

## Start the Podman container

```bash
podman run --rm \
  --publish 127.0.0.1:9714:9714 \
  duui-speaker-anonymization:1.0
```

Health endpoint: `http://127.0.0.1:9714/v1/health`

## Run within DUUI

The input CAS sofa must contain a base64-encoded audio file. Configure a target
view for the transcript:

```java
composer.add(
    new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
        .withParameter("language", "de")
        .withTargetView("transcript")
        .build()
);
```

If `language` is omitted, the component uses the CAS document language and
defaults to English when it is unspecified.

## Input and Output

| Direction | CAS sofa/view | MIME type | Description |
|---|---|---|---|
| Input | Source sofa | `audio/wav` recommended | Base64-encoded audio file |
| Output | Configured target view | `text/plain` | Original transcript |
| Output | `opf_anonymized_audio` | `audio/wav` | Base64-encoded anonymized WAV |

Malformed audio and unsupported languages return HTTP 400. Decoded input is
limited to 100 MiB by default; configure it with
`DUUI_SPEAKER_ANONYMIZATION_MAX_AUDIO_BYTES`.

# Build

From the `duui-speaker-anonymization` directory:

```bash
podman build \
  --format docker \
  --tag duui-speaker-anonymization:1.0 \
  --file src/main/docker/Dockerfile .
```

The build downloads and verifies the IMS-Toucan v2.5 checkpoints, Whisper
large-v3, and Silero VAD. Model revisions and SHA-256 digests are pinned, and
the resulting image runs offline as an unprivileged user. Expect an image of
approximately 11 GB.

For direct Python development, create the Conda environment from
`environment.yml` and download the model files with:

```bash
python src/main/python/models/download_models.py \
  --destination src/main/python/models
```

Large model files are intentionally excluded from Git.

# Tests

With the component running on port 9714:

```bash
mvn test
```

The integration tests process German and English WAV fixtures and validate the
transcripts and anonymized RIFF/WAVE output. Set
`DUUI_SPEAKER_ANONYMIZATION_URL` to test another endpoint.

# Credits and License

This component uses
[IMS speaker-anonymization](https://github.com/DigitalPhonetics/speaker-anonymization/),
[IMS-Toucan](https://github.com/DigitalPhonetics/IMS-Toucan), and
[Whisper large-v3](https://huggingface.co/openai/whisper-large-v3). See the
[Interspeech 2024 paper](https://doi.org/10.21437/Interspeech.2024-1615) for the
underlying anonymization work.

The component is distributed under the repository's
[AGPL-3.0 license](../LICENSE). Bundled IMS-Toucan code retains its
[upstream license](src/main/python/anonymization/modules/tts/IMSToucan/LICENSE).

# Cite

If you use this component, please cite DUUI:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler.
(2023). “Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI.”
Findings of the Association for Computational Linguistics: EMNLP 2023,
385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)]
[[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

```bibtex
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399}
}

@misc{wolf2026duuispeakeranonymization,
  author       = {Wolf, Tim},
  title        = {Speaker Anonymization as {DUUI} Component},
  year         = {2026},
  howpublished = {\url{https://github.com/texttechnologylab/duui-uima/tree/main/duui-speaker-anonymization}}
}
```
