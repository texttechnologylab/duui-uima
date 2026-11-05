# Text Easability PCA: CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, 4724 excerpts; duui-coh-metrix 0.6.0, default pipeline

Created 2026-11-04T21:21:11+00:00 by `easability/pca.py` from `/storage/projects/CORE/projects2/coh_metrix/clear_corpus/indices_en_0.6.0.csv`; model `src/main/resources/easability/text_easability_en_clear.json`. Method: Graesser, McNamara & Kulikowich (2011), see the docstring of `pca.py`.

## How to read this report

- **Loading**: correlation of an input measure with a component (−1 to +1), oriented so that a higher component score means easier text. Loadings with |l| ≥ .40 are usually read as relevant.
- **Explained variance**: share of the variance of all input measures that a component accounts for; compare with the published share (Graesser et al. 2011, TASA).
- **Marker score**: mean loading of a name's marker measures (`components.csv`, sign-adjusted so that positive = easier). About 1 = the markers load strongly on this component, about 0 = not at all. Every name is assigned to exactly one component (maximal sum of |marker score|), so a name is assigned even if no component fits it; a low marker score or a higher score on another component means the name does not fit.
- **r Lexile**: Spearman correlation of the component score with the Lexile band (higher = harder), comparable with the published correlation with DRP grade level (higher = harder). A negative value means the component is higher (easier) for easier texts. **r BT** (teacher-rated easiness, higher = easier) is expected with the opposite sign.
- **Literary / informational**: mean z score per CLEAR genre (the reference corpus has mean 0, SD 1).

## Data

- Documents: 4724, of which 23 with at least one missing input value (replaced by the corpus mean).
- Input measures: 49 of the 53 published measures.
- Not available in Coh-Metrix 3.0: 9 Ratio of function words to content words, 41 Lexical diversity—verbs, 52 Verb tense repetition, 53 Verb aspect repetition.
- Excluded: .

| Measure | Missing documents | Share |
|---|---|---|
| 44 LSA overlap—all sentences | 23 | 0.5% |

## Components compared with the original

Total explained variance of the 8 components: 63.3% (published: 67.3%).

| Component | Varimax component | Marker score | Higher marker score elsewhere | Variance | Published variance | r Lexile | Published r DRP | r BT | Literary | Informational |
|---|---|---|---|---|---|---|---|---|---|---|
| PCNARz Narrativity | 1 | 0.60 | – | 12.2% | 18.5% | -0.39 | -0.69 | +0.25 | +0.48 | -0.50 |
| PCSYNz Syntactic simplicity | 7 | 0.36 | varimax 3: 0.41 | 6.6% | 9.5% | -0.46 | -0.47 | +0.22 | +0.04 | -0.05 |
| PCCNCz Word concreteness | 4 | 0.79 | – | 6.3% | 6.3% | -0.19 | -0.23 | +0.27 | +0.36 | -0.38 |
| PCREFz Referential cohesion | 2 | 0.82 | – | 10.4% | 14.1% | +0.22 | +0.03 | +0.05 | -0.15 | +0.16 |
| PCDCz Deep cohesion | 6 | 0.89 | – | 3.9% | 5.5% | -0.05 | +0.11 | +0.02 | +0.01 | -0.01 |
| PCVERBz Verb cohesion | 8 | 0.71 | – | 8.6% | 5.4% | -0.41 | – | +0.46 | +0.06 | -0.06 |
| PCCONNz Connectivity | 5 | 0.68 | – | 9.2% | 4.1% | -0.14 | – | +0.19 | +0.37 | -0.39 |
| PCTEMPz Temporality | 3 | 0.09 | varimax 5: 0.28 | 6.0% | 4.0% | -0.25 | – | +0.08 | -0.06 | +0.07 |

## Strongest loadings per component

**PCNARz Narrativity** (Varimax component 1; markers: 6:+;8:+;16:+;3:+;2:-;1:-;19:-)

| Measure | Loading |
|---|---|
| 6 Pronouns | +0.86 |
| 16 Word frequency (logarithm) | +0.72 |
| 2 Nouns | -0.70 |
| 17 Content word frequency (logarithm) | +0.64 |
| 7 First-person pronouns | +0.57 |
| 29 Modifiers per noun phrase | -0.56 |
| 8 Third-person pronouns | +0.56 |
| 1 Syllables per word | -0.53 |
| 24 Negations | +0.52 |
| 18 Minimum word frequency per sentence | +0.51 |
| 19 Age of acquisition | -0.49 |
| 23 Familiarity | +0.46 |
| 31 Passive constructions | -0.46 |
| 37 Noun overlap—adjacent sentences | -0.43 |
| 5 Adverbs | +0.33 |
| 15 Adversative connectives | +0.33 |
| 42 LSA-givenness versus newness | +0.33 |
| 3 Verbs | +0.31 |

**PCSYNz Syntactic simplicity** (Varimax component 7; markers: 28:-;29:-;30:-;45:-;32:+)

| Measure | Loading |
|---|---|
| 43 LSA overlap—adjacent sentences | -0.82 |
| 44 LSA overlap—all sentences | -0.81 |
| 28 Words per sentence | -0.61 |
| 30 Words before main verb of main clause | -0.50 |
| 32 Syntactic similarity—sentences in paragraph | +0.41 |
| 29 Modifiers per noun phrase | -0.39 |
| 39 Type-token ratio | -0.36 |
| 3 Verbs | +0.35 |
| 18 Minimum word frequency per sentence | +0.32 |

**PCCNCz Word concreteness** (Varimax component 4; markers: 21:+;22:+;20:+)

| Measure | Loading |
|---|---|
| 22 Imagery | +0.89 |
| 21 Concreteness | +0.78 |
| 20 Meaningfulness | +0.70 |
| 19 Age of acquisition | -0.50 |
| 1 Syllables per word | -0.48 |
| 27 Polysemy—multiple senses of word | +0.31 |

**PCREFz Referential cohesion** (Varimax component 2; markers: 33:+;34:+;35:+;36:+;37:+;38:+)

| Measure | Loading |
|---|---|
| 38 Stem overlap—all sentences | +0.89 |
| 36 Argument overlap—all sentences | +0.88 |
| 35 Argument overlap—adjacent sentences | +0.86 |
| 34 Content word overlap—all sentences | +0.84 |
| 33 Content word overlap—adjacent sentences | +0.77 |
| 37 Noun overlap—adjacent sentences | +0.70 |
| 40 Lexical diversity—all words | -0.42 |
| 39 Type-token ratio | -0.40 |
| 43 LSA overlap—adjacent sentences | +0.36 |
| 28 Words per sentence | +0.35 |
| 44 LSA overlap—all sentences | +0.32 |

**PCDCz Deep cohesion** (Varimax component 6; markers: 11:+;47:+)

| Measure | Loading |
|---|---|
| 47 Causal cohesion | +0.94 |
| 11 Causal connectives | +0.84 |
| 25 Causal verbs | -0.34 |

**PCVERBz Verb cohesion** (Varimax component 8; markers: 49:+;50:+)

| Measure | Loading |
|---|---|
| 50 LSA verb overlap—adjacent sentences | +0.73 |
| 49 Verb overlap—adjacent | +0.68 |
| 17 Content word frequency (logarithm) | +0.61 |
| 23 Familiarity | +0.56 |
| 18 Minimum word frequency per sentence | +0.56 |
| 39 Type-token ratio | -0.54 |
| 27 Polysemy—multiple senses of word | +0.53 |
| 19 Age of acquisition | -0.51 |
| 42 LSA-givenness versus newness | +0.46 |
| 16 Word frequency (logarithm) | +0.45 |
| 25 Causal verbs | +0.45 |
| 1 Syllables per word | -0.35 |

**PCCONNz Connectivity** (Varimax component 5; markers: 13:+;14:+;15:+;10:+)

| Measure | Loading |
|---|---|
| 13 Logical connectives | +0.87 |
| 14 Additive connectives | +0.86 |
| 10 Connectives | +0.84 |
| 3 Verbs | +0.51 |
| 27 Polysemy—multiple senses of word | +0.44 |
| 2 Nouns | -0.43 |
| 5 Adverbs | +0.43 |
| 12 Temporal connectives | +0.42 |
| 1 Syllables per word | -0.41 |
| 8 Third-person pronouns | +0.40 |
| 4 Adjectives | -0.38 |
| 19 Age of acquisition | -0.36 |

**PCTEMPz Temporality** (Varimax component 3; markers: 51:+;12:+)

| Measure | Loading |
|---|---|
| 46 Dissimilarity of words between sentences | -0.79 |
| 45 Dissimilarity of parts of speech between sentences | -0.78 |
| 32 Syntactic similarity—sentences in paragraph | +0.68 |
| 42 LSA-givenness versus newness | +0.53 |
| 28 Words per sentence | -0.42 |
| 40 Lexical diversity—all words | -0.41 |
| 39 Type-token ratio | -0.39 |
| 33 Content word overlap—adjacent sentences | +0.33 |

## Marker scores: every name on every Varimax component

Sign-adjusted mean marker loading before orienting the components; the assignment uses the absolute values. Bold: assigned component.

| Name | Varimax 1 | Varimax 2 | Varimax 3 | Varimax 4 | Varimax 5 | Varimax 6 | Varimax 7 | Varimax 8 |
|---|---|---|---|---|---|---|---|---|
| Narrativity | **+0.60** | -0.05 | -0.02 | +0.19 | -0.38 | +0.02 | -0.03 | +0.26 |
| Syntactic simplicity | +0.15 | -0.05 | -0.41 | +0.02 | -0.04 | -0.03 | **-0.36** | +0.13 |
| Word concreteness | -0.07 | -0.05 | -0.10 | **+0.79** | -0.03 | -0.04 | -0.13 | +0.00 |
| Referential cohesion | -0.04 | **+0.82** | -0.07 | -0.04 | -0.03 | +0.03 | +0.13 | +0.04 |
| Deep cohesion | +0.08 | +0.07 | +0.02 | -0.07 | -0.09 | **+0.89** | +0.05 | +0.02 |
| Verb cohesion | +0.11 | +0.15 | -0.10 | -0.05 | -0.01 | +0.01 | -0.10 | **+0.71** |
| Connectivity | +0.19 | +0.05 | +0.06 | +0.01 | **-0.68** | +0.10 | +0.03 | -0.01 |
| Temporality | -0.08 | +0.07 | **-0.09** | +0.04 | -0.28 | +0.03 | +0.02 | +0.06 |

## Eigenvalues (unrotated PCA)

Components with eigenvalue > 1 (Kaiser criterion): 12; 8 are kept as in the original.

| Component | Eigenvalue | Explained variance | Cumulative |
|---|---|---|---|
| 1 | 10.29 | 21.0% | 21.0% |
| 2 | 5.98 | 12.2% | 33.2% |
| 3 | 5.10 | 10.4% | 43.6% |
| 4 | 2.95 | 6.0% | 49.6% |
| 5 | 1.93 | 3.9% | 53.6% |
| 6 | 1.77 | 3.6% | 57.2% |
| 7 | 1.57 | 3.2% | 60.4% |
| 8 | 1.42 | 2.9% | 63.3% |
| 9 | 1.32 | 2.7% | 66.0% |
| 10 | 1.14 | 2.3% | 68.3% |
| 11 | 1.07 | 2.2% | 70.5% |
| 12 | 1.00 | 2.0% | 72.5% |
| 13 | 0.98 | 2.0% | 74.5% |
| 14 | 0.96 | 2.0% | 76.5% |
| 15 | 0.89 | 1.8% | 78.3% |

## Input measure pairs with |r| > 0.9

Removed by Graesser et al. (2011) before their PCA; kept here, the published measures are used unchanged.

| Measure | Measure | r |
|---|---|---|
| 13 Logical connectives | 14 Additive connectives | +0.97 |
| 36 Argument overlap—all sentences | 38 Stem overlap—all sentences | +0.99 |
| 43 LSA overlap—adjacent sentences | 44 LSA overlap—all sentences | +0.92 |
