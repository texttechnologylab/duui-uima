# Text Easability PCA: Reference corpus v1 (easability/REFERENCE_PLAN.md): 9 registers x 1,000 passages of 250-350 words cut by TASA_de's rule (news report, personal narrative, encyclopedia, informational description, research and expert, how-to, opinion/review/advice, persuasion/commercial from FineWeb-2 and TASA_de's sources; fiction 1900-1930 from de-Corp); Coh-Metrix 0.6.0-dev annotation (snapshot code_20261104_v060dev), German index values identical in 0.6.0-py and 0.6.1-py (README "Version 0.6.1-py")

Created 2026-11-05T07:55:00+00:00 by `easability/build_model.py` of coh-metrix_resurrection (adapted from `pca.py`, duui-coh-metrix commit 75f97bf) from `/data/CORE/export/t0/coh_metrix/reference_v1/cohmetrix_v060dev_20261104`; model `src/main/resources/easability/text_easability_de_refv1-ud-rr-cr5.json` (model id `refv1-ud-rr-cr5`). Written in `pca.py`'s layout on 2026-11-19T15:05:14+00:00 by `easability/export_duui_results.py`; nothing was refitted. Method: Graesser, McNamara & Kulikowich (2011), as in `pca.py`; the differences are listed at the end.

## How to read this report

- **Loading**: correlation of an input measure with a component (−1 to +1), oriented so that a higher component score means easier text. Loadings with |l| ≥ .40 are usually read as relevant.
- **Explained variance**: share of the variance of all input measures that a component accounts for; compare with the published share (Graesser et al. 2011, TASA).
- **Marker score**: mean loading of a name's marker measures (`components.csv`, sign-adjusted so that positive = easier). About 1 = the markers load strongly on this component, about 0 = not at all. Every name is assigned to exactly one component (maximal sum of |marker score|), so a name is assigned even if no component fits it; a low marker score or a higher score on another component means the name does not fit.
- **r Lexile**: Spearman correlation of the component score with the Lexile band (higher = harder), comparable with the published correlation with DRP grade level (higher = harder). A negative value means the component is higher (easier) for easier texts. **r BT** (teacher-rated easiness, higher = easier) is expected with the opposite sign. The reference corpus has no ratings, so both are computed on CLEAR's test half (2359 texts, in German translation) scored with this model.
- **Literary / informational**: mean z score per CLEAR genre on the same texts. The z scores are relative to the reference corpus (mean 0, SD 1 there), so these means show where CLEAR's excerpts lie on it.

## Data

- Documents: 9000 (encyclopedia 1000; fiction 1000; how-to 1000; informational description 1000; news report 1000; opinion, review, advice 1000; personal narrative 1000; persuasion, commercial 1000; research and expert 1000), of which 2 with at least one missing input value (replaced by the corpus mean).
- Input measures: 45 of the 53 published measures.
- Not available in Coh-Metrix 3.0: 9 Ratio of function words to content words, 41 Lexical diversity—verbs, 52 Verb tense repetition, 53 Verb aspect repetition.
- Excluded: 48 Intentional cohesion (excluded since 2026-11-04 by the user's decision: as an input it formed a component of its own and merged Narrativity and Connectivity).
- Removed as redundant (Graesser et al. 2011 removed one measure of each pair with |r| > 0.9; here the one with the higher mean |r| with the remaining measures): 38 Stem overlap—all sentences, 13 Logical connectives, 43 LSA overlap—adjacent sentences.
- Validation texts: CLEAR's test half (the seeded split of 2026-10-28), 2361 texts, in the Kolibri-1 translation that TTLab's `clear_de_kolibri-1-q4km_pca` model was fitted on (46 flagged translations kept; the ratings belong to the English originals); 2 with undefined components (an input missing that the reference corpus never misses) left out.

| Measure | Missing documents | Share |
|---|---|---|
| 44 LSA overlap—all sentences | 1 | 0.0% |
| 49 Verb overlap—adjacent | 1 | 0.0% |
| 50 LSA verb overlap—adjacent sentences | 1 | 0.0% |

## Components compared with the original

Total explained variance of the 8 components: 65.1% (published: 67.3%).

| Component | Varimax component | Marker score | Higher marker score elsewhere | Variance | Published variance | r Lexile | Published r DRP | r BT | Literary | Informational |
|---|---|---|---|---|---|---|---|---|---|---|
| PCNARz Narrativity | 1 | 0.62 | – | 18.2% | 18.5% | -0.42 | -0.69 | +0.29 | +0.56 | -0.26 |
| PCSYNz Syntactic simplicity | 4 | 0.48 | – | 7.2% | 9.5% | -0.62 | -0.47 | +0.29 | -0.30 | -0.26 |
| PCCNCz Word concreteness | 3 | 0.74 | – | 7.4% | 6.3% | -0.30 | -0.23 | +0.39 | +1.76 | +0.45 |
| PCREFz Referential cohesion | 2 | 0.81 | – | 11.6% | 14.1% | -0.09 | +0.03 | +0.21 | +0.63 | +0.69 |
| PCDCz Deep cohesion | 5 | 0.85 | – | 4.2% | 5.5% | +0.05 | +0.11 | -0.03 | -0.10 | -0.17 |
| PCVERBz Semantic overlap | 8 | 0.22 | varimax 2: 0.32 | 5.0% | 5.4% | -0.15 | – | +0.16 | -0.11 | -0.24 |
| PCCONNz Connectivity | 6 | 0.62 | – | 6.6% | 4.1% | -0.01 | – | +0.14 | +0.41 | -0.31 |
| PCTEMPz Third-person narration | 7 | 0.27 | – | 4.8% | 4.0% | +0.01 | – | +0.07 | +1.70 | +0.53 |

## Strongest loadings per component

**PCNARz Narrativity** (Varimax component 1; markers: 6:+;8:+;16:+;3:+;2:-;1:-;19:-)

| Measure | Loading |
|---|---|
| 17 Content word frequency (logarithm) | +0.85 |
| 16 Word frequency (logarithm) | +0.82 |
| 6 Pronouns | +0.77 |
| 7 First-person pronouns | +0.73 |
| 2 Nouns | -0.73 |
| 18 Minimum word frequency per sentence | +0.72 |
| 27 Polysemy—multiple senses of word | +0.66 |
| 1 Syllables per word | -0.63 |
| 5 Adverbs | +0.60 |
| 29 Modifiers per noun phrase | -0.57 |
| 19 Age of acquisition | -0.57 |
| 24 Negations | +0.53 |
| 23 Familiarity | +0.52 |
| 3 Verbs | +0.52 |
| 50 LSA verb overlap—adjacent sentences | -0.49 |
| 26 Intentional actions, events, and particles | +0.46 |
| 31 Passive constructions | -0.44 |
| 10 Connectives | +0.41 |
| 51 Temporal cohesion | -0.36 |
| 37 Noun overlap—adjacent sentences | -0.34 |
| 8 Third-person pronouns | +0.34 |
| 45 Dissimilarity of parts of speech between sentences | +0.32 |
| 15 Adversative connectives | +0.32 |

**PCSYNz Syntactic simplicity** (Varimax component 4; markers: 28:-;29:-;30:-;45:-;32:+)

| Measure | Loading |
|---|---|
| 28 Words per sentence | -0.82 |
| 32 Syntactic similarity—sentences in paragraph | +0.64 |
| 44 LSA overlap—all sentences | -0.60 |
| 30 Words before main verb of main clause | -0.58 |
| 18 Minimum word frequency per sentence | +0.50 |
| 3 Verbs | +0.36 |
| 36 Argument overlap—all sentences | -0.35 |
| 4 Adjectives | -0.31 |
| 29 Modifiers per noun phrase | -0.31 |

**PCCNCz Word concreteness** (Varimax component 3; markers: 21:+;22:+;20:+)

| Measure | Loading |
|---|---|
| 22 Imagery | +0.88 |
| 21 Concreteness | +0.86 |
| 19 Age of acquisition | -0.64 |
| 1 Syllables per word | -0.51 |
| 20 Meaningfulness | +0.48 |
| 42 LSA-givenness versus newness | -0.37 |
| 44 LSA overlap—all sentences | -0.30 |

**PCREFz Referential cohesion** (Varimax component 2; markers: 33:+;34:+;35:+;36:+;37:+;38:+)

| Measure | Loading |
|---|---|
| 34 Content word overlap—all sentences | +0.86 |
| 33 Content word overlap—adjacent sentences | +0.85 |
| 35 Argument overlap—adjacent sentences | +0.84 |
| 36 Argument overlap—all sentences | +0.82 |
| 40 Lexical diversity—all words | -0.70 |
| 39 Type-token ratio | -0.70 |
| 37 Noun overlap—adjacent sentences | +0.67 |
| 4 Adjectives | -0.42 |
| 46 Dissimilarity of words between sentences | -0.40 |
| 49 Verb overlap—adjacent | +0.34 |
| 50 LSA verb overlap—adjacent sentences | +0.30 |

**PCDCz Deep cohesion** (Varimax component 5; markers: 11:+;47:+)

| Measure | Loading |
|---|---|
| 47 Causal cohesion | +0.94 |
| 11 Causal connectives | +0.75 |
| 25 Causal verbs | -0.49 |

**PCVERBz Semantic overlap** (Varimax component 8; markers: 49:+;50:+)

| Measure | Loading |
|---|---|
| 45 Dissimilarity of parts of speech between sentences | -0.71 |
| 42 LSA-givenness versus newness | +0.68 |
| 46 Dissimilarity of words between sentences | -0.53 |
| 44 LSA overlap—all sentences | +0.43 |
| 32 Syntactic similarity—sentences in paragraph | +0.39 |
| 50 LSA verb overlap—adjacent sentences | +0.36 |
| 4 Adjectives | +0.34 |

**PCCONNz Connectivity** (Varimax component 6; markers: 13:+;14:+;15:+;10:+)

| Measure | Loading |
|---|---|
| 10 Connectives | +0.78 |
| 14 Additive connectives | +0.64 |
| 5 Adverbs | +0.58 |
| 15 Adversative connectives | +0.45 |
| 2 Nouns | -0.45 |
| 20 Meaningfulness | -0.44 |
| 17 Content word frequency (logarithm) | +0.35 |
| 3 Verbs | +0.34 |
| 25 Causal verbs | +0.30 |
| 11 Causal connectives | +0.30 |

**PCTEMPz Third-person narration** (Varimax component 7; markers: 51:+;12:+)

| Measure | Loading |
|---|---|
| 8 Third-person pronouns | +0.66 |
| 12 Temporal connectives | +0.46 |
| 6 Pronouns | +0.46 |
| 3 Verbs | +0.46 |
| 29 Modifiers per noun phrase | -0.42 |
| 49 Verb overlap—adjacent | -0.39 |
| 39 Type-token ratio | +0.36 |

## Marker scores: every name on every Varimax component

Sign-adjusted mean marker loading before orienting the components; the assignment uses the absolute values. Bold: assigned component.

| Name | Varimax 1 | Varimax 2 | Varimax 3 | Varimax 4 | Varimax 5 | Varimax 6 | Varimax 7 | Varimax 8 |
|---|---|---|---|---|---|---|---|---|
| Narrativity | **+0.62** | +0.07 | +0.20 | -0.18 | -0.01 | +0.23 | +0.28 | -0.00 |
| Syntactic simplicity | +0.17 | +0.01 | +0.05 | **-0.48** | -0.02 | +0.02 | +0.07 | +0.19 |
| Word concreteness | +0.06 | -0.02 | **+0.74** | -0.07 | -0.09 | -0.17 | +0.12 | -0.01 |
| Referential cohesion | +0.01 | **+0.81** | -0.02 | +0.19 | +0.03 | -0.01 | +0.10 | +0.06 |
| Deep cohesion | +0.11 | +0.02 | -0.10 | +0.04 | **+0.85** | +0.19 | +0.02 | +0.04 |
| Semantic overlap | -0.23 | +0.32 | -0.02 | -0.08 | -0.08 | +0.02 | -0.24 | **+0.22** |
| Connectivity | +0.33 | +0.00 | -0.01 | -0.01 | +0.08 | **+0.62** | +0.12 | -0.03 |
| Third-person narration | -0.14 | +0.12 | +0.17 | -0.10 | -0.03 | +0.21 | **+0.27** | +0.08 |

## Eigenvalues (unrotated PCA)

Components with eigenvalue > 1 (Kaiser criterion): 10; parallel analysis (200 random data sets, eigenvalue above the 95th percentile): 9; 8 are kept as in the original.

| Component | Eigenvalue | Explained variance | Cumulative |
|---|---|---|---|
| 1 | 11.23 | 24.9% | 24.9% |
| 2 | 5.64 | 12.5% | 37.5% |
| 3 | 3.85 | 8.6% | 46.0% |
| 4 | 2.57 | 5.7% | 51.8% |
| 5 | 1.77 | 3.9% | 55.7% |
| 6 | 1.60 | 3.6% | 59.3% |
| 7 | 1.42 | 3.2% | 62.4% |
| 8 | 1.18 | 2.6% | 65.0% |
| 9 | 1.12 | 2.5% | 67.5% |
| 10 | 1.05 | 2.3% | 69.9% |
| 11 | 0.92 | 2.0% | 71.9% |
| 12 | 0.90 | 2.0% | 73.9% |
| 13 | 0.84 | 1.9% | 75.8% |
| 14 | 0.83 | 1.8% | 77.6% |
| 15 | 0.79 | 1.7% | 79.4% |

## Input measure pairs with |r| > 0.9

Graesser et al. (2011) removed one measure of each such pair before their PCA; so does this model (removed: 38 Stem overlap—all sentences, 13 Logical connectives, 43 LSA overlap—adjacent sentences).

| Measure | Measure | r |
|---|---|---|
| 13 Logical connectives | 14 Additive connectives | +0.98 |
| 36 Argument overlap—all sentences | 38 Stem overlap—all sentences | +1.00 |
| 43 LSA overlap—adjacent sentences | 44 LSA overlap—all sentences | +0.93 |

## Differences from duui-coh-metrix's pca.py

- **Reference corpus**: not CLEAR. CLEAR is the criterion these models are validated on, so it is not also their reference; the reference corpora are built to cover what people read (9 registers × 1,000 passages of 250-350 words, one passage per document).
- **Index values**: Coh-Metrix-py (coh-metrix_resurrection), as named in the title, not duui-coh-metrix's default pipeline.
- **Inputs**: measure 48 excluded (above); the redundancy rule removes one measure of each pair with |r| > .90 (pca.py keeps all).
- **Names**: PCVERBz is called Semantic overlap (Coh-Metrix 3.0: Verb cohesion; it carries LSA overlap and givenness), PCTEMPz Third-person narration (Coh-Metrix 3.0: Temporality; third-person pronouns, little verb overlap); labels and values are those of the 3.0 components.
- **Validation**: on texts the model was not fitted on (CLEAR's test half); pca.py's validation uses its own reference corpus.
- **Further checks** of this model (parallel analysis in full, bootstrap stability of the names, fits per register, anchor correlations) are in build_model.py's report: `/data/CORE/export/t0/coh_metrix/easability/refv1_ud_rr_cr5/report.md`.
