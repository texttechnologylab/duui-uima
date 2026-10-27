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
  `ClearCorpusTextEasabilityTest` use the local build
  `duui-coh-metrix:easability-dev` of the working tree; replace it with the
  released image (and the spaCy remote URL with its Docker image) when the
  image is released.
- Old labels in existing results: `paper_eval/results` (`_gptlist`,
  `_wiki10000`, ...) and the paper evaluation data in Postgres. Rename or
  recompute before comparing with new results.
