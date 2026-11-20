# Text Easability PCA: Reference corpus A, English (easability/REFERENCE_PLAN_EN.md): 9 registers x 1,000 passages of 250-350 words cut by TASA_de's rule with English settings (news report, personal narrative, encyclopedia, informational description, research and expert, how-to, opinion/review/advice, persuasion/commercial from FineWeb sample-10BT, wiki40b, Wikinews and Intereconomics; fiction 1900-1918 from PG-19); Coh-Metrix annotation with the fixed ELECTRA-large tokenizer (snapshot code_20261104_v060, assets/hf with vocab.txt / tokenizer.json / tokenizer_config.json, request AG of 2026-11-05; README "Version 0.6.1-py")

Created 2026-11-05T08:38:42+00:00 by `easability/build_model.py` of coh-metrix_resurrection (adapted from `pca.py`, duui-coh-metrix commit 75f97bf) from `/data/CORE/export/t0/coh_metrix/reference_en_a/cohmetrix_v061en_20261105`; model `src/main/resources/easability/text_easability_en_refa-en-rr-gn-cr5.json` (model id `refa-en-rr-gn-cr5`). Written in `pca.py`'s layout on 2026-11-19T15:05:14+00:00 by `easability/export_duui_results.py`; nothing was refitted. Method: Graesser, McNamara & Kulikowich (2011), as in `pca.py`; the differences are listed at the end.

## How to read this report

- **Loading**: correlation of an input measure with a component (−1 to +1), oriented so that a higher component score means easier text. Loadings with |l| ≥ .40 are usually read as relevant.
- **Explained variance**: share of the variance of all input measures that a component accounts for; compare with the published share (Graesser et al. 2011, TASA).
- **Marker score**: mean loading of a name's marker measures (`components.csv`, sign-adjusted so that positive = easier). About 1 = the markers load strongly on this component, about 0 = not at all. Here the names are not assigned by the marker scores but taken from the German model (see the end); the marker scores show how well each name's markers fit the component that carries it.
- **r Lexile**: Spearman correlation of the component score with the Lexile band (higher = harder), comparable with the published correlation with DRP grade level (higher = harder). A negative value means the component is higher (easier) for easier texts. **r BT** (teacher-rated easiness, higher = easier) is expected with the opposite sign. The reference corpus has no ratings, so both are computed on CLEAR's test half (2361 texts) scored with this model.
- **Literary / informational**: mean z score per CLEAR genre on the same texts. The z scores are relative to the reference corpus (mean 0, SD 1 there), so these means show where CLEAR's excerpts lie on it.

## Data

- Documents: 9000 (encyclopedia 1000; fiction 1000; how-to 1000; informational description 1000; news report 1000; opinion, review, advice 1000; personal narrative 1000; persuasion, commercial 1000; research and expert 1000), of which 51 with at least one missing input value (replaced by the corpus mean).
- Input measures: 44 of the 53 published measures.
- Not available in Coh-Metrix 3.0: 9 Ratio of function words to content words, 41 Lexical diversity—verbs, 52 Verb tense repetition, 53 Verb aspect repetition.
- Excluded: 48 Intentional cohesion (excluded since 2026-11-04 by the user's decision: as an input it formed a component of its own and merged Narrativity and Connectivity).
- Removed as redundant (Graesser et al. 2011 removed one measure of each pair with |r| > 0.9; here the one with the higher mean |r| with the remaining measures): 38 Stem overlap—all sentences, 13 Logical connectives, 16 Word frequency (logarithm), 43 LSA overlap—adjacent sentences.
- Validation texts: CLEAR's test half (the seeded split of 2026-10-28), 2361 texts; 0 with undefined components (an input missing that the reference corpus never misses) left out.

| Measure | Missing documents | Share |
|---|---|---|
| 44 LSA overlap—all sentences | 50 | 0.6% |
| 51 Temporal cohesion | 2 | 0.0% |

## Components compared with the original

Total explained variance of the 8 components: 65.8% (published: 67.3%).

| Component | Varimax component | Marker score | Higher marker score elsewhere | Variance | Published variance | r Lexile | Published r DRP | r BT | Literary | Informational |
|---|---|---|---|---|---|---|---|---|---|---|
| PCNARz Narrativity | 1 | 0.68 | – | 20.8% | 18.5% | -0.60 | -0.69 | +0.50 | +0.76 | -0.29 |
| PCSYNz Syntactic simplicity | 8 | 0.23 | varimax 4: 0.41 | 4.5% | 9.5% | -0.43 | -0.47 | +0.17 | +0.02 | +0.00 |
| PCCNCz Word concreteness | 3 | 0.80 | – | 6.6% | 6.3% | -0.21 | -0.23 | +0.32 | +1.12 | +0.21 |
| PCREFz Referential cohesion | 2 | 0.82 | – | 11.2% | 14.1% | +0.09 | +0.03 | +0.14 | +0.16 | +0.28 |
| PCDCz Deep cohesion | 7 | 0.86 | – | 4.2% | 5.5% | +0.00 | +0.11 | -0.01 | +0.22 | +0.08 |
| PCVERBz Semantic overlap | 4 | 0.01 | varimax 5: 0.60 | 5.6% | 5.4% | -0.36 | – | +0.14 | -0.06 | +0.16 |
| PCCONNz Connectivity | 6 | 0.54 | – | 7.9% | 4.1% | -0.04 | – | +0.14 | +0.99 | +0.00 |
| PCTEMPz Third-person narration | 5 | 0.14 | varimax 6: 0.41 | 5.0% | 4.0% | +0.26 | – | -0.26 | +0.57 | +0.06 |

## Strongest loadings per component

**PCNARz Narrativity** (Varimax component 1; markers: 6:+;8:+;16:+;3:+;2:-;1:-;19:-)

| Measure | Loading |
|---|---|
| 17 Content word frequency (logarithm) | +0.89 |
| 6 Pronouns | +0.87 |
| 18 Minimum word frequency per sentence | +0.80 |
| 19 Age of acquisition | -0.77 |
| 1 Syllables per word | -0.76 |
| 2 Nouns | -0.75 |
| 23 Familiarity | +0.74 |
| 29 Modifiers per noun phrase | -0.69 |
| 7 First-person pronouns | +0.68 |
| 50 LSA verb overlap—adjacent sentences | +0.60 |
| 27 Polysemy—multiple senses of word | +0.60 |
| 24 Negations | +0.58 |
| 37 Noun overlap—adjacent sentences | -0.56 |
| 8 Third-person pronouns | +0.54 |
| 42 LSA-givenness versus newness | +0.49 |
| 5 Adverbs | +0.48 |
| 3 Verbs | +0.42 |
| 28 Words per sentence | -0.39 |
| 4 Adjectives | -0.38 |
| 31 Passive constructions | -0.38 |
| 30 Words before main verb of main clause | -0.35 |
| 26 Intentional actions, events, and particles | +0.35 |
| 49 Verb overlap—adjacent | +0.35 |
| 10 Connectives | +0.33 |

**PCSYNz Syntactic simplicity** (Varimax component 8; markers: 28:-;29:-;30:-;45:-;32:+)

| Measure | Loading |
|---|---|
| 44 LSA overlap—all sentences | -0.73 |
| 28 Words per sentence | -0.45 |
| 4 Adjectives | -0.41 |
| 30 Words before main verb of main clause | -0.40 |
| 42 LSA-givenness versus newness | -0.36 |
| 32 Syntactic similarity—sentences in paragraph | +0.36 |
| 45 Dissimilarity of parts of speech between sentences | +0.31 |

**PCCNCz Word concreteness** (Varimax component 3; markers: 21:+;22:+;20:+)

| Measure | Loading |
|---|---|
| 22 Imagery | +0.90 |
| 21 Concreteness | +0.84 |
| 20 Meaningfulness | +0.64 |
| 19 Age of acquisition | -0.39 |

**PCREFz Referential cohesion** (Varimax component 2; markers: 33:+;34:+;35:+;36:+;37:+;38:+)

| Measure | Loading |
|---|---|
| 34 Content word overlap—all sentences | +0.88 |
| 36 Argument overlap—all sentences | +0.85 |
| 33 Content word overlap—adjacent sentences | +0.85 |
| 35 Argument overlap—adjacent sentences | +0.85 |
| 37 Noun overlap—adjacent sentences | +0.67 |
| 39 Type-token ratio | -0.59 |
| 40 Lexical diversity—all words | -0.57 |
| 28 Words per sentence | +0.40 |
| 44 LSA overlap—all sentences | +0.39 |
| 46 Dissimilarity of words between sentences | -0.32 |

**PCDCz Deep cohesion** (Varimax component 7; markers: 11:+;47:+)

| Measure | Loading |
|---|---|
| 47 Causal cohesion | +0.93 |
| 11 Causal connectives | +0.80 |
| 25 Causal verbs | -0.34 |

**PCVERBz Semantic overlap** (Varimax component 4; markers: 49:+;50:+)

| Measure | Loading |
|---|---|
| 46 Dissimilarity of words between sentences | -0.76 |
| 45 Dissimilarity of parts of speech between sentences | -0.74 |
| 32 Syntactic similarity—sentences in paragraph | +0.71 |
| 28 Words per sentence | -0.42 |
| 40 Lexical diversity—all words | -0.39 |
| 42 LSA-givenness versus newness | +0.34 |

**PCCONNz Connectivity** (Varimax component 6; markers: 13:+;14:+;15:+;10:+)

| Measure | Loading |
|---|---|
| 10 Connectives | +0.79 |
| 14 Additive connectives | +0.62 |
| 12 Temporal connectives | +0.62 |
| 3 Verbs | +0.59 |
| 27 Polysemy—multiple senses of word | +0.42 |
| 2 Nouns | -0.39 |
| 4 Adjectives | -0.38 |
| 5 Adverbs | +0.38 |
| 1 Syllables per word | -0.38 |
| 25 Causal verbs | +0.36 |
| 19 Age of acquisition | -0.32 |
| 20 Meaningfulness | -0.30 |

**PCTEMPz Third-person narration** (Varimax component 5; markers: 51:+;12:+)

| Measure | Loading |
|---|---|
| 49 Verb overlap—adjacent | -0.67 |
| 50 LSA verb overlap—adjacent sentences | -0.53 |
| 39 Type-token ratio | +0.49 |
| 8 Third-person pronouns | +0.45 |
| 25 Causal verbs | -0.42 |
| 27 Polysemy—multiple senses of word | -0.37 |

## Marker scores: every name on every Varimax component

Sign-adjusted mean marker loading before orienting the components. Bold: the component that carries the name (assigned by congruence with the German model, not by these scores).

| Name | Varimax 1 | Varimax 2 | Varimax 3 | Varimax 4 | Varimax 5 | Varimax 6 | Varimax 7 | Varimax 8 |
|---|---|---|---|---|---|---|---|---|
| Narrativity | **+0.68** | -0.07 | +0.03 | -0.04 | -0.04 | +0.37 | +0.01 | +0.03 |
| Syntactic simplicity | +0.27 | -0.11 | +0.00 | -0.41 | +0.05 | +0.11 | -0.00 | **+0.23** |
| Word concreteness | -0.05 | -0.06 | **+0.80** | -0.06 | -0.08 | -0.04 | -0.08 | +0.07 |
| Referential cohesion | -0.13 | **+0.82** | -0.03 | -0.03 | +0.03 | +0.00 | +0.01 | -0.12 |
| Deep cohesion | +0.11 | +0.05 | -0.11 | +0.03 | +0.00 | +0.06 | **+0.86** | -0.01 |
| Semantic overlap | +0.48 | +0.18 | -0.14 | **-0.01** | +0.60 | -0.01 | -0.00 | +0.02 |
| Connectivity | +0.26 | -0.01 | -0.08 | +0.03 | -0.05 | **+0.54** | +0.16 | -0.08 |
| Third-person narration | -0.07 | +0.02 | +0.10 | -0.11 | **+0.14** | +0.41 | -0.01 | +0.07 |

## Eigenvalues (unrotated PCA)

Components with eigenvalue > 1 (Kaiser criterion): 10; parallel analysis (200 random data sets, eigenvalue above the 95th percentile): 9; 8 are kept as in the original.

| Component | Eigenvalue | Explained variance | Cumulative |
|---|---|---|---|
| 1 | 11.34 | 25.8% | 25.8% |
| 2 | 5.46 | 12.4% | 38.2% |
| 3 | 3.44 | 7.8% | 46.0% |
| 4 | 2.26 | 5.1% | 51.1% |
| 5 | 1.93 | 4.4% | 55.5% |
| 6 | 1.65 | 3.8% | 59.3% |
| 7 | 1.57 | 3.6% | 62.8% |
| 8 | 1.29 | 2.9% | 65.8% |
| 9 | 1.09 | 2.5% | 68.3% |
| 10 | 1.03 | 2.3% | 70.6% |
| 11 | 0.93 | 2.1% | 72.7% |
| 12 | 0.90 | 2.0% | 74.7% |
| 13 | 0.88 | 2.0% | 76.7% |
| 14 | 0.84 | 1.9% | 78.6% |
| 15 | 0.78 | 1.8% | 80.4% |

## Input measure pairs with |r| > 0.9

Graesser et al. (2011) removed one measure of each such pair before their PCA; so does this model (removed: 38 Stem overlap—all sentences, 13 Logical connectives, 16 Word frequency (logarithm), 43 LSA overlap—adjacent sentences).

| Measure | Measure | r |
|---|---|---|
| 13 Logical connectives | 14 Additive connectives | +0.96 |
| 16 Word frequency (logarithm) | 17 Content word frequency (logarithm) | +0.93 |
| 36 Argument overlap—all sentences | 38 Stem overlap—all sentences | +0.99 |
| 43 LSA overlap—adjacent sentences | 44 LSA overlap—all sentences | +0.90 |

## Differences from duui-coh-metrix's pca.py

- **Reference corpus**: not CLEAR. CLEAR is the criterion these models are validated on, so it is not also their reference; the reference corpora are built to cover what people read (9 registers × 1,000 passages of 250-350 words, one passage per document).
- **Index values**: Coh-Metrix-py (coh-metrix_resurrection), as named in the title, not duui-coh-metrix's default pipeline.
- **Inputs**: measure 48 excluded (above); the redundancy rule removes one measure of each pair with |r| > .90 (pca.py keeps all).
- **Names**: PCVERBz is called Semantic overlap (Coh-Metrix 3.0: Verb cohesion; it carries LSA overlap and givenness), PCTEMPz Third-person narration (Coh-Metrix 3.0: Temporality; third-person pronouns, little verb overlap); labels and values are those of the 3.0 components.
- **Naming by congruence**: the components are named and oriented by Tucker's congruence with the German model `refv1-ud-rr-cr5` over 44 shared measures (one-to-one, summed |phi| maximal), so the two languages' components correspond; the marker scores are reported but not used.
- **Validation**: on texts the model was not fitted on (CLEAR's test half); pca.py's validation uses its own reference corpus.
- **Further checks** of this model (parallel analysis in full, bootstrap stability of the names, fits per register, anchor correlations) are in build_model.py's report: `/data/CORE/export/t0/coh_metrix/easability/refa_en_rr_gn_cr5/report.md`.
