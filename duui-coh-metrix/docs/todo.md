# Coh-Metrix: open questions and todos

## Labels: questions for the original implementer

Details and current labels: `docs/label-open-questions.md`.

1. **`DRNEG_llmlist`**: where does the negation lemma list come from
   (`_DE_NEGATION_LEMMAS`, `_EN_NEGATION_LEMMAS`)? The implementation
   documentation only says "konservative sprachspezifische Lemmaliste". If it
   is hand-written, the `_llmlist` suffix is wrong.
2. **Causal particles (`SMCAUSvp`, `SMCAUSr`)**: were the particle lists
   generated with an LLM, like the intentional particles? Then both get
   `_llmlist`.
3. **Verb classes (`SMCAUSv`, `SMCAUSvp`, `SMINTEp`, `SMCAUSr`, currently
   `_wordnet`)**: who chose the WordNet synsets (`cause.v.01`, `intend.v.01`,
   `plan.v.01`) and the German seed verbs? If the selection counts as an
   implementation detail (the original also classifies verbs with WordNet),
   the public labels lose the suffix.
4. **`RDFKGL_textstat`**: does the textstat fork change the formula for German,
   and which syllable counter does it use? Only to document; the label stays
   (README rule 10).

## Documentation

- Update the implementation documentation (`docs/Coh_Metrix Voll.pdf`) to the
  new label scheme. It still shows the old labels (`_germanet`/`_wordnet`
  public labels, `WRDFRQ*_wiki10000`, CELEX outputs as `None`, `_gptlist`), no
  `label_internal` and no index version.
- Decide which files in `docs/` go into the repository. The README refers to
  `docs/coh-metrix-30-indices.md`, which is not committed; the paper PDFs and
  Markdown copies may not be publishable.

## Text Easability (in development, not committed)

Full list: `easability/OPEN_QUESTIONS.md`, section "Index labels".

- Decide the PC label names before committing: `_clear` only, or rule 3
  (inherit the input suffixes)? Internal suffix for the German model:
  `_cleartranslate` does not fit if the German reference corpus is not a
  translation of CLEAR.
- Commit together: the PC label rename `_wikipedia` -> `_clear`,
  `cm_text_easability` and its loop in `post_process`, the README section and
  `_clear` suffix row, `.gitignore`, `pom.xml`, the Dockerfile, and the
  validation test changes.
- Rerun the CLEAR pipeline before the next PCA: the processed XMIs carry the
  old labels.
- Three components do not replicate (Syntactic simplicity, Verb cohesion,
  Temporality): set them to NaN or change the assignment.
- `SMINTEp` probably undercounts intentional verbs (none in 71% of the CLEAR
  excerpts).

## Performance

- `SMCAUSlsa` dominates the runtime: on a synthetic 500-sentence document (300-dimensional vectors) it takes
  5.8 of 7.9 s, because `get_SMCAUSlsa` compares all verb pairs in a Python double loop. Vectorize it like the
  all-pairs cosine of the LSA indices (one normalized matrix product), with an equivalence test against the
  current loop. Text Easability depends on it as an input. Excluding `SMCAUSlsa` (`exclude_indices`) avoids
  the cost in the meantime.
- Measure the WordNet/GermaNet-based indices (`WRDPOLc`, `WRDHYP*`, `SMCAUSwn`) on real text; the synthetic
  benchmark words are not in the lexicons.

## Tests and data

- `CohMetrixDockerValidationTest` pins the image `duui-coh-metrix:0.1.2`; it
  fails on the new labels until a new image is built and the version is
  updated. Its "intentionally unimplemented" PC check must be replaced once
  Text Easability is committed.
- Old labels in existing results: `paper_eval/results` (`_gptlist`,
  `_wiki10000`, ...) and the paper evaluation data in Postgres. Rename or
  recompute before comparing with new results.
