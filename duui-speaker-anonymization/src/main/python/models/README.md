# Model artifacts

The five IMS-Toucan v2.5 checkpoints are downloaded and checksum-verified by
`download_models.py`. They are intentionally excluded from Git because two of
the files exceed GitHub's regular 100 MiB object limit.

To populate or verify a local checkout:

```bash
python src/main/python/models/download_models.py \
  --destination src/main/python/models

python src/main/python/models/download_models.py \
  --destination src/main/python/models \
  --verify-only
```

The container build performs the same verified download. Whisper large-v3 and
Silero VAD are pinned and cached separately during the image build.

