# Video Anonymization DUUI

DUUI component for anonymizing faces and voices in MP4 or WebM videos. It combines
[`duui-face_anon`](../duui-face_anon) and
[`duui-speaker-anonymization`](../duui-speaker-anonymization). Each request to
`duui-video-anon` coordinates four stages in Python. Java `DUUIComposer`
instances call only the face and speaker services:

1. Anonymize faces in the video.
2. Extract its audio as a WAV file.
3. Anonymize the speaker's voice.
4. Replace the video's audio with the anonymized voice.

Face redaction supports blur, blackout, and pixelation. Generative face
anonymization is also available with a Hugging Face token. The speaker component
supports English, German, French, Italian, Spanish, Portuguese, Dutch, Polish,
and Russian.

## Parameters

| Setting | Default | Description |
|---|---|---|
| `anon_type` | `redact` | `redact`, `single_align`, or `multiple_align` |
| `redact_type` | `black` | `blur`, `black`, or `pixel` when redacting faces |
| `hf_token` | empty | Required for generative face anonymization |
| `language` | `en` | Language passed to speaker anonymization |

Pass these as DUUI component parameters. Environment variables `DUUI_FACE_MODE`,
`DUUI_REDACT_TYPE`, `HF_TOKEN`, and `DUUI_LANGUAGE` supply defaults. Set
`DUUI_FACE_ANON_URL` and `DUUI_SPEAKER_ANON_URL` on the container to the URLs of
the running services. `MAX_MEDIA_BYTES` defaults to 500 MiB.

The runner sets face sampling to `uniform` with `frame_interval=1` to keep the
video timeline aligned with the audio. See the face component's README for its
additional model settings.

Both service URLs are required HTTP(S) URLs. The video service validates them
at startup and exits immediately if either is missing or malformed.

## How To Use

This component requires running face and speaker DUUI services. The speaker
container needs its model files. Face redaction does not need an HF token.

### Build and start the component

From the `duui-video-anon` directory:

```bash
docker build -t duui-video-anon -f src/main/docker/Dockerfile .
docker run --rm -p 9715:9715 \
  -e DUUI_FACE_ANON_URL=http://anduin.hucompute.org:42927 \
  -e DUUI_SPEAKER_ANON_URL=http://anduin.hucompute.org:38455 \
  duui-video-anon
```

The component exposes the standard DUUI `/v1/typesystem`,
`/v1/communication_layer`, `/v1/documentation`, and `/v1/process` endpoints.
Its health endpoint is `/v1/health`.

### Logging

The component uses [DUUIlogger](https://github.com/texttechnologylab/DUUIlogger)
for structured request, processing stage, and error logs. Logs appear in the
container console as processing runs. To collect them in an external DUUI
composer, enable a debug level:

```java
composer.withDebugLevel(DUUIComposer.DebugLevel.INFO);
```

A DUUI driver with component logging support sends `DUUI-Log-Collect: true`.
The component then returns that request's logs in the `DUUI-Logs` response
header when processing finishes, including logs from its Python worker thread.
The header is capped at 16,000 bytes by DUUIlogger. Requests without collection
still log to the console. Media payloads and request options are not included
in progress messages.

### Use within DUUI

Add only this component to an external `DUUIComposer` with a JSON Lua context.
The [Java integration test](src/test/java/org/texttechnologylab/duui/videoanon/VideoAnonComponentTest.java)
contains a complete example:

```java
composer.add(new DUUIRemoteDriver.Component("http://anduin.hucompute.org:9715")
    .withParameter("anon_type", "redact")
    .withParameter("redact_type", "black")
    .withParameter("language", "en")
    .withTargetView("output")
    .build().withTimeout(7200));
```

## Input and Output

| Direction | CAS view | Data |
|---|---|---|
| Input | `_InitialView` | Base64 MP4 or WebM in `org.texttechnologylab.annotation.type.Video.src` |
| Output | `output` | Base64 MP4 in a `Video` annotation |

Python passes the face result to `extract_audio()` and the speaker result to
`mux_audio()` directly. The Java speaker stage accepts either the
`anonymized_audio` or `opf_anonymized_audio` CAS view, since some speaker images
write to the latter.

The component extracts the first audio stream as a 16 kHz mono WAV. It maps only
the face-anonymized video and new audio into the output, pads or trims the new
audio to the video duration, and never copies the original voice. Silent videos
remain silent. If the speaker result is empty for a video with audio, muxing
fails. Resynthesized speech can differ from the source video in word timing.

Media is Base64 encoded throughout the pipeline, so the composer and services
need enough memory for the video and audio data.

## Tests

Run the local pipeline orchestration tests with:

```bash
python3 -m pip install -r requirements-test.txt
python3 -m unittest discover -s src/test/python
```

The concurrency test sends two `/v1/process` requests through the ASGI app
while limiting its worker thread pool to two tokens. Both requests must finish
without calling the video service from inside the pipeline.

The Java integration test sends `src/test/resources/videos/hope.webm` to the
single video component and saves `target/test-output/video-anonymized.mp4`.
With all three services running, enable it with:

```bash
DUUI_RUN_INTEGRATION=true mvn -q -Dtest=VideoAnonComponentTest test
```

For a shorter run, pass `-Dduui.video.input=/path/to/short.mp4`. Override the
video component URL with `DUUI_VIDEO_ANON_URL` or `-Dduui.video.url=...`.

## Cite

If you use this component, please cite DUUI:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler.
(2023). “Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI.”
Findings of the Association for Computational Linguistics: EMNLP 2023,
385–399. [Paper](https://aclanthology.org/2023.findings-emnlp.29).

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

@misc{wolf2026duuivideoanon,
  author       = {Wolf, Tim},
  title        = {Video Anonymization as {DUUI} Component},
  year         = {2026},
  howpublished = {\url{https://github.com/texttechnologylab/duui-uima/tree/main/duui-video-anon}}
}
```

This component is distributed under the repository's [AGPL-3.0 license](../LICENSE).
