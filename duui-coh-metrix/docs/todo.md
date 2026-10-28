# Coh-Metrix: open questions and todos

## Labels

Two-layer scheme since 0.2.0 (`docs/label-scheme.md`, README "Index labels").

- **Verb classes** (`SMCAUSv`, `SMCAUSvp`, `SMINTEp`, `SMCAUSr`, `SMINTEr`):
  who chose the WordNet synsets (`cause.v.01`, `intend.v.01`, `plan.v.01`)
  and the German seed verbs? The original implementer will check. Only to
  document; the labels name the resource (`WN`, `GN`, `SEEDlist`) either way.
- **Own vectors**: `vectors` accepts only `SPACY` so far; design the input of
  user-supplied embedding annotations (README "Vectors").
- The German labels with `GN` were tested only with the `SEEDlist` fallback
  (no GermaNet in the test image); check once with GermaNet mounted.

## Documentation

- Update the implementation documentation (`docs/Coh_Metrix Voll.pdf`) to the
  two-layer label scheme (`docs/label-scheme.md`). It still shows the labels of
  0.1.1 (`_germanet`/`_wordnet`, `WRDFRQ*_wiki10000`, CELEX outputs as `None`,
  `_gptlist`) and no index version.
- Decide which files in `docs/` go into the repository. The README refers to
  `docs/coh-metrix-30-indices.md`, which is not committed; the paper PDFs and
  Markdown copies may not be publishable.

## Text Easability (English first test committed, German in development)

Full list: `easability/OPEN_QUESTIONS.md`, section "Index labels".

- German model from the first translation (Kolibri-1 Q4_K_M) done; redo
  pipeline, PCA and comparison when the translation changes.
- Three components do not replicate (Syntactic simplicity, Verb cohesion,
  Temporality): set them to NaN or change the assignment. Syntactic simplicity
  and Verb cohesion are named the other way round in English and German, and
  Temporality has the opposite sign (`easability/OPEN_QUESTIONS.md`).
- `SMINTEp` probably undercounts intentional verbs (none in 71% of the CLEAR
  excerpts).

## CLEAR translation (German model)

First version: all 4724 excerpts translated by Kolibri-1 Q4_K_M
(`translation_eval/translate_clear.py`, `translation_eval/kolibri/Dockerfile`)
into `/storage/projects/CORE/projects2/coh_metrix/clear_corpus/translation_kolibri-1-q4km/`,
log `translation_kolibri-1-q4km.log` next to it. WMT24++ data and Kolibri
translations (`translation_eval/translate_wmt24pp.py`):
`/storage/projects/CORE/projects2/coh_metrix/translation_eval/data/`. Used as
is for now; open:

- **Two broken translations**: 2589 and 4500 ran into the 32k-token limit
  (`finish_reason: length`) in a repetition loop (19,067 and 23,673 words
  instead of about 190). The script writes them like normal results.
  Retranslate (per paragraph or with `max_tokens`) or exclude; let the script
  treat `length` as an error.
- **54 paragraph mismatches** (1.1%, `PARAGRAPH MISMATCH` in the log, includes
  4500): mostly ±1 paragraph, some far off (4621 14→5, 7211 14→9, 6531 11→7,
  7147 3→1); about a dozen single-paragraph excerpts split into 2–3.
  Paragraph-based indices need the same count. Retranslate per paragraph
  (WMT25 fallback, Kocmi et al. 2025) or exclude.
- **Possibly missing content**: 4 excerpts with fewer than 75% of the source
  words (median 95%): 6011, 7221, 4445, 6472. 6011 drops a whole clause and
  writes "one thousand seven hundred and eighty nine" as "1789"; the other
  three are not checked. A dropped sentence is not caught by the paragraph
  check.
- **Model choice**: compare with two other models on WMT24++
  (`translation_eval/load_wmt24pp.py`), including Coh-Metrix indices of
  source, reference and translations. Kolibri-1 knowledge cutoff (June 2026)
  is after WMT24++, so contamination is possible.
- **Server**: the Kolibri container accepts requests without the API key
  (`LLAMA_ARG_API_KEY` not set at start).

## Performance

- `SMCAUSlsa` dominates the runtime: on a synthetic 500-sentence document (300-dimensional vectors) it takes
  5.8 of 7.9 s, because `get_SMCAUSlsa` compares all verb pairs in a Python double loop. Vectorize it like the
  all-pairs cosine of the LSA indices (one normalized matrix product), with an equivalence test against the
  current loop. Text Easability depends on it as an input. Excluding `SMCAUSlsa` (`exclude_indices`) avoids
  the cost in the meantime.
- Measure the WordNet/GermaNet-based indices (`WRDPOLc`, `WRDHYP*`, `SMCAUSwn`) on real text; the synthetic
  benchmark words are not in the lexicons.

## Tests and data

- `CohMetrixDockerValidationTest`, `ClearCorpusCohMetrixTest` and
  `ClearCorpusTextEasabilityTest` use the image
  `docker.texttechnologylab.org/duui-coh-metrix:0.2.0`
  (`src/main/bash/docker_build.sh`); run them once it is built. Replace the
  spaCy remote URL in `ClearCorpusCohMetrixTest` with its Docker image.
- Old labels in existing results: `paper_eval/results` (`_gptlist`,
  `_wiki10000`, `_spacy`, ...), the paper evaluation data in Postgres, and
  the processed CLEAR XMIs (`easability/extract_indices.py` now reads
  `labelV3` and needs XMIs of 0.2.0). Rename or recompute before comparing
  with new results.
