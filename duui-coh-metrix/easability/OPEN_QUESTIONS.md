# Text Easability: open questions

Recorded 2026-10-23 during the first CLEAR (English) pilot. Not blocking the first test implementation.

## Decisions taken for the first test

- Measure 32 uses `SYNSTRUTa`, measure 48 is excluded (`pca_input_measures.csv`); 48 of 53 measures remain.
- Remaining missing values: reference-corpus mean (815 of 4724 excerpts, almost all measure 47); at runtime a
  missing input counts as the reference mean as well.
- |r| > .90 pairs are kept (the published 53 measures contain them): logical/additive connectives (.91),
  concreteness/imagery (.92), argument/stem overlap all sentences (.99), LSA adjacent/all sentences (.91).
- Component names and directions come from marker measures (`components.csv`) derived from the component
  descriptions in McNamara et al. (2014, ch. 5), not from Table 1 of Graesser et al. (2011), which we do not
  have. With Table 1 the markers should be replaced by its primary loadings.
- Component scores: rotated PCA score weights (the original method is not published); percentile: empirical
  distribution in CLEAR (1001 quantiles).

## First test result: three components do not replicate

`results/clear_en_pca/` (written by `pca.py`; 60.4% variance explained, published 67.3%). Clear matches: Narrativity, Word
concreteness, Referential cohesion, Connectivity, Deep cohesion. Not matched (marker score in `assignment.csv`):

- **Syntactic simplicity (0.42)** is assigned to Varimax component 3, sentence-to-sentence similarity (MED,
  LSA given/new). Words per sentence (−.71) and words before the main verb (−.51) load on Varimax component 8
  together with LSA overlap; its Syntactic simplicity marker score is 0.31. Component 8 has the strongest
  relationship to difficulty (r Lexile −.59), as Syntactic simplicity in the original, but is assigned to Verb
  cohesion. Syntactic simplicity and Verb cohesion are probably swapped.
- **Verb cohesion (0.25)**: no verb overlap component; the Verb cohesion markers score about the same on Varimax
  components 1, 3 and 8 (|0.25–0.27|).
- **Temporality (0.00)**: no temporal component (highest marker score 0.15); the assigned component is word
  frequency. Every name is assigned to exactly one component, so the name is assigned anyway.

All marker scores: `results/clear_en_pca/marker_scores.csv` and `report.md`. To decide: set the three indices
to NaN until they replicate, or change the assignment.

Possible causes: short excerpts (173 vs. 289 words), spaCy-vector LSA, the excluded/mapped measures, CLEAR
instead of TASA. Validation (`validation.csv`, Spearman with the Lexile band vs. published r with DRP):
Narrativity −.48 (−.69), Syntactic simplicity −.18 (−.47), Word concreteness −.14 (−.23), Referential cohesion
.21 (.03), Deep cohesion .06 (.11).

## German model (2026-10-27): CLEAR translated with Kolibri-1 Q4_K_M

`results/clear_de_kolibri-1-q4km_pca/` (4669 of 4724 excerpts: the reader skips 2 truncated translations and 54
paragraph mismatches, see `docs/todo.md`; GermaNet V180; 58.5% variance explained). Comparison with English:
`results/clear_en_vs_de_kolibri-1-q4km/report.md`.

- **Carried over**: Narrativity (congruence of the loadings .88, correlation of the scores of the same excerpts
  .90), Word concreteness (.94, .71), Referential cohesion (.97, .71), Connectivity (.97, .87). Validation similar
  to English (r Lexile: Narrativity −.53, Word concreteness −.18, Referential cohesion +.18).
- **Syntactic simplicity and Verb cohesion are swapped between the languages**: the same two components exist in
  both (sentence-to-sentence similarity: MED, LSA given/new, SYNSTRUTa; sentence length with LSA overlap), but
  the marker assignment names them differently: English Syntactic simplicity = German Verb cohesion (congruence
  .93), English Verb cohesion = German Syntactic simplicity (.90). In German, Syntactic simplicity is the
  sentence-length component (r Lexile −.49, published −.47), which matches the original; in English it is not
  (see above). So PCSYN and PCVERB are not comparable across languages until the assignment is fixed, e.g. by
  assigning Syntactic simplicity by words per sentence in both languages.
- **Temporality** has the opposite sign (congruence −.79): it is the word frequency component in both, and its
  marker score is about 0, so the orientation is arbitrary.
- **Deep cohesion** differs (congruence .64, r .30).
- Missing inputs: causal cohesion undefined in 20.3% (English 16.8%).

## Index labels (change from another session)

The new label scheme (README, "Index labels") is committed (a8dd58b), but without Text Easability. In that commit the
PC indices still have the old placeholder labels (`PCNARz_wikipedia`, ..., "Not implemented").

TODO when committing Text Easability:

- Commit the PC label rename `_wikipedia` -> `_clear` in `duui.py` (16 entries) together with `cm_text_easability`,
  the `post_process` loop that sets `label_internal` from the model's `label_suffix`, and `import json`.
- README: the "Text Easability reference corpus" section and the `_clear` row of the suffix table.
- `CohMetrixDockerValidationTest`: normalize `_clear` instead of `_wikipedia`, and replace the "intentionally
  unimplemented" PC check (it expects NaN and "Not implemented").
- `.gitignore`, `pom.xml` (commons-csv), Dockerfile (`COPY src/main/resources/easability`).
- Decide the PC label names first: `_clear` only (current), or rule 3 (inherit the input suffixes, see below)? And
  `_cleartranslate` as the internal suffix for a German model built from translated CLEAR, or a different corpus
  name if the German reference corpus is not a translation.

- `pca_input_measures.csv` and `text_easability_en.json` were renamed to the new public labels
  (`WRDFRQa_wiki10000` -> `WRDFRQa_wiki`, `WRDAOAc_mrctranslate` -> `WRDAOAc`, `WRDPOLc_wordnet` -> `WRDPOLc`,
  `SMCAUSwn_wordnet` -> `SMCAUSwn`); all 50 model input labels exist in `INDEX_CATALOG`, and no old labels are
  left in the easability scripts and Java tests.
- Checked 2026-10-26: CLEAR pipeline rerun with a local build of the working tree
  (`duui-coh-metrix:easability-dev`, PC group excluded so the reference corpus does not depend on an earlier
  model). The input values are identical to the 0.1.2 run for all 4724 excerpts, including `LSAGN_spacy` (the SVD
  fix does not change CLEAR); only the labels differ. Extraction and PCA rerun: the result is unchanged (scores
  differ by < 3e-14). Old data kept in `clear_corpus/xmi_en_0.1.2/` and `clear_corpus/indices_en_0.1.2.csv`
  (not committed).
- Checked 2026-10-26: `ClearCorpusTextEasabilityTest` also runs with `include_indices=PCNARz,PCSYNp`: only these
  two indices are output, with values identical to the full run (48 excerpts).
- Label of the PC indices, decided 2026-10-26: public label `label_v3` + `_clear` for all languages (rule 3 is
  not applied); `label_internal` from the model's `label_suffix`: `_clear` for English, `_cleartranslate` for
  the German model (translated CLEAR, in progress).
- Undefined values, decided 2026-10-26: like the other indices, the PC indices are None (no error) when they
  cannot really be calculated. A missing input counts as the reference mean only if it is missing in CLEAR as
  well (`missing_documents` > 0 in the model: measures 43, 44, 47, 51); any other missing input makes all PC
  indices None. Before, an empty document gave z 0 / p ~50; all other indices were already None for empty
  documents (EN and DE). On the 143 English validation fixtures the PC indices are defined for 26; the others
  are short test texts with a single sentence (no sentence pair, 68), no MRC age-of-acquisition word
  (`WRDAOAc`, 73) or fewer than two verbs (`SMCAUSwn`, `SMCAUSlsa_spacy`, 69). None of these inputs is missing
  in CLEAR.
- Tests (EN): `CohMetrixDockerValidationTest` uses the local image `duui-coh-metrix:easability-dev` (TODO:
  released image); the "Not implemented" placeholder check is replaced: the empty document must give NaN
  without error for all indices including the PC indices, and `textEasabilityEnglish` recomputes the PC indices
  of every English fixture from its input indices and the model (z within 1e-9, NaN if undefined, p in
  [0, 100]). German: no model yet, the PC indices have the error "No Text Easability model"; add tests with
  the German model.
- The 16 PC placeholder blocks in `post_process` still contain the old "Not implemented" error; it is
  overwritten at the end of `post_process` (for a language without model by "No Text Easability model").

## Missing values in the PCA input measures

Full CLEAR corpus (4724 excerpts), share of documents with a missing value:

| Measure (2011) | Index | Missing | Cause |
|---|---|---|---|
| 48 Intentional cohesion | `SMINTEr_llmlist` | 71.0% | ratio undefined without intentional verbs (`SMINTEp_wordnet` = 0 in 71.0%) |
| 32 Syntactic similarity—sentences in paragraph | `SYNSTRUTt` | 34.8% | across paragraphs, undefined for single-paragraph excerpts |
| 47 Causal cohesion | `SMCAUSr_wordnet` | 16.8% | ratio undefined without causal verbs |
| 44 LSA overlap—all sentences | `LSASSp_spacy` | 0.6% | |

Listwise deletion keeps only a small, biased subset (286 of the first 1641 documents). Options:

1. Measure 32: use `SYNSTRUTa` (adjacent sentences, no missing values) instead of `SYNSTRUTt`?
2. Measure 48: exclude until the intentional verb count is checked (see below)?
3. Remaining missing values (mainly measure 47): mean imputation (neutral after standardization, report counts)
   or ratio = 0 without causal verbs (changes the index definition)?

## Intentional verbs (`SMINTEp`) probably undercounted

`duui.py` (`_causal_practical_verbs_intentional`, English) uses only the WordNet hyponyms of `intend.v.01` and
`plan.v.01`. McNamara et al. (2014, ch. 4) give "contact, drop, walk, talk" as examples of intentional verbs
(actions of animate agents). In CLEAR (about half fiction), 71% of the excerpts have no intentional verb.
Check against the original definition; affects measures 26 and 48.

## Further choices in `pca_input_measures.csv`

- 39 Type-token ratio: `LDTTRc` (content words) or `LDTTRa` (all words)?
- 40 Lexical diversity—all words: `LDMTLDa` (MTLD) or `LDVOCDa` (VOCD)?
- 9, 41, 52, 53 not available in Coh-Metrix 3.0 (function/content word ratio, verb lexical diversity, separate
  tense and aspect repetition).
- |r| > .90 pairs (removed by Graesser et al. 2011 before the PCA) are only reported, not removed:
  concreteness/imagery (.92), argument overlap all/stem overlap all (.98) in the partial run.
