# Text Easability PCA: CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, 4724 excerpts

Created 2026-10-26T12:31:15+00:00 by `easability/pca.py` from `clear_corpus/indices_en.csv`; model `src/main/resources/easability/text_easability_en.json`. Method: Graesser, McNamara & Kulikowich (2011), see the docstring of `pca.py`.

## How to read this report

- **Loading**: correlation of an input measure with a component (−1 to +1), oriented so that a higher component score means easier text. Loadings with |l| ≥ .40 are usually read as relevant.
- **Explained variance**: share of the variance of all input measures that a component accounts for; compare with the published share (Graesser et al. 2011, TASA).
- **Marker score**: mean loading of a name's marker measures (`components.csv`, sign-adjusted so that positive = easier). About 1 = the markers load strongly on this component, about 0 = not at all. Every name is assigned to exactly one component (maximal sum of |marker score|), so a name is assigned even if no component fits it; a low marker score or a higher score on another component means the name does not fit.
- **r Lexile**: Spearman correlation of the component score with the Lexile band (higher = harder), comparable with the published correlation with DRP grade level (higher = harder). A negative value means the component is higher (easier) for easier texts. **r BT** (teacher-rated easiness, higher = easier) is expected with the opposite sign.
- **Literary / informational**: mean z score per CLEAR genre (the reference corpus has mean 0, SD 1).

## Data

- Documents: 4724, of which 815 with at least one missing input value (replaced by the corpus mean).
- Input measures: 48 of the 53 published measures.
- Not available in Coh-Metrix 3.0: 9 Ratio of function words to content words, 41 Lexical diversity—verbs, 52 Verb tense repetition, 53 Verb aspect repetition.
- Excluded: 48 Intentional cohesion (intentional verbs probably undercounted, undefined for 71% of CLEAR (see OPEN_QUESTIONS.md)).

| Measure | Missing documents | Share |
|---|---|---|
| 43 LSA overlap—adjacent sentences | 2 | 0.0% |
| 44 LSA overlap—all sentences | 28 | 0.6% |
| 47 Causal cohesion | 793 | 16.8% |
| 51 Temporal cohesion | 2 | 0.0% |

## Components compared with the original

Total explained variance of the 8 components: 60.4% (published: 67.3%).

| Component | Varimax component | Marker score | Higher marker score elsewhere | Variance | Published variance | r Lexile | Published r DRP | r BT | Literary | Informational |
|---|---|---|---|---|---|---|---|---|---|---|
| PCNARz Narrativity | 1 | 0.63 | – | 13.4% | 18.5% | -0.48 | -0.69 | +0.38 | +0.65 | -0.68 |
| PCSYNz Syntactic simplicity | 3 | 0.42 | – | 7.3% | 9.5% | -0.18 | -0.47 | +0.08 | -0.06 | +0.06 |
| PCCNCz Word concreteness | 4 | 0.84 | – | 6.6% | 6.3% | -0.14 | -0.23 | +0.21 | +0.24 | -0.25 |
| PCREFz Referential cohesion | 2 | 0.83 | – | 10.8% | 14.1% | +0.21 | +0.03 | +0.08 | -0.16 | +0.17 |
| PCDCz Deep cohesion | 7 | 0.64 | – | 3.9% | 5.5% | +0.06 | +0.11 | -0.07 | +0.03 | -0.04 |
| PCVERBz Verb cohesion | 8 | 0.25 | varimax 3: 0.27 | 8.7% | 5.4% | -0.59 | – | +0.44 | +0.00 | -0.00 |
| PCCONNz Connectivity | 5 | 0.68 | – | 5.8% | 4.1% | +0.06 | – | +0.04 | +0.14 | -0.15 |
| PCTEMPz Temporality | 6 | 0.00 | varimax 3: 0.15 | 3.9% | 4.0% | +0.02 | – | -0.06 | +0.07 | -0.08 |

## Strongest loadings per component

**PCNARz Narrativity** (Varimax component 1; markers: 6:+;8:+;16:+;3:+;2:-;1:-;19:-)

| Measure | Loading |
|---|---|
| 6 Pronouns | +0.87 |
| 2 Nouns | -0.80 |
| 1 Syllables per word | -0.76 |
| 29 Modifiers per noun phrase | -0.75 |
| 8 Third-person pronouns | +0.72 |
| 27 Polysemy—multiple senses of word | +0.60 |
| 19 Age of acquisition | -0.60 |
| 3 Verbs | +0.59 |
| 4 Adjectives | -0.51 |
| 50 LSA verb overlap—adjacent sentences | +0.47 |
| 31 Passive constructions | -0.46 |
| 23 Familiarity | +0.43 |
| 24 Negations | +0.43 |
| 5 Adverbs | +0.42 |
| 7 First-person pronouns | +0.42 |
| 37 Noun overlap—adjacent sentences | -0.37 |
| 15 Adversative connectives | +0.35 |

**PCSYNz Syntactic simplicity** (Varimax component 3; markers: 28:-;29:-;30:-;45:-;32:+)

| Measure | Loading |
|---|---|
| 46 Dissimilarity of words between sentences | -0.79 |
| 45 Dissimilarity of parts of speech between sentences | -0.76 |
| 42 LSA-givenness versus newness | +0.75 |
| 32 Syntactic similarity—sentences in paragraph | +0.64 |
| 39 Type-token ratio | -0.42 |
| 40 Lexical diversity—all words | -0.42 |
| 28 Words per sentence | -0.36 |
| 51 Temporal cohesion | +0.31 |
| 50 LSA verb overlap—adjacent sentences | +0.30 |

**PCCNCz Word concreteness** (Varimax component 4; markers: 21:+;22:+;20:+)

| Measure | Loading |
|---|---|
| 22 Imagery | +0.92 |
| 21 Concreteness | +0.86 |
| 20 Meaningfulness | +0.74 |
| 19 Age of acquisition | -0.49 |
| 1 Syllables per word | -0.32 |
| 5 Adverbs | -0.32 |
| 11 Causal connectives | -0.31 |

**PCREFz Referential cohesion** (Varimax component 2; markers: 33:+;34:+;35:+;36:+;37:+;38:+)

| Measure | Loading |
|---|---|
| 38 Stem overlap—all sentences | +0.87 |
| 36 Argument overlap—all sentences | +0.86 |
| 34 Content word overlap—all sentences | +0.85 |
| 35 Argument overlap—adjacent sentences | +0.85 |
| 33 Content word overlap—adjacent sentences | +0.79 |
| 37 Noun overlap—adjacent sentences | +0.73 |
| 39 Type-token ratio | -0.46 |
| 40 Lexical diversity—all words | -0.42 |
| 43 LSA overlap—adjacent sentences | +0.36 |
| 28 Words per sentence | +0.35 |
| 44 LSA overlap—all sentences | +0.32 |

**PCDCz Deep cohesion** (Varimax component 7; markers: 11:+;47:+)

| Measure | Loading |
|---|---|
| 47 Causal cohesion | +0.72 |
| 25 Causal verbs | -0.59 |
| 11 Causal connectives | +0.55 |
| 3 Verbs | -0.36 |
| 5 Adverbs | +0.33 |
| 4 Adjectives | +0.31 |

**PCVERBz Verb cohesion** (Varimax component 8; markers: 49:+;50:+)

| Measure | Loading |
|---|---|
| 43 LSA overlap—adjacent sentences | -0.77 |
| 44 LSA overlap—all sentences | -0.76 |
| 28 Words per sentence | -0.71 |
| 18 Minimum word frequency per sentence | +0.63 |
| 39 Type-token ratio | -0.61 |
| 30 Words before main verb of main clause | -0.51 |
| 32 Syntactic similarity—sentences in paragraph | +0.41 |
| 23 Familiarity | +0.41 |
| 49 Verb overlap—adjacent | +0.40 |
| 3 Verbs | +0.32 |
| 36 Argument overlap—all sentences | -0.32 |
| 38 Stem overlap—all sentences | -0.31 |

**PCCONNz Connectivity** (Varimax component 5; markers: 13:+;14:+;15:+;10:+)

| Measure | Loading |
|---|---|
| 13 Logical connectives | +0.94 |
| 14 Additive connectives | +0.92 |
| 10 Connectives | +0.83 |

**PCTEMPz Temporality** (Varimax component 6; markers: 51:+;12:+)

| Measure | Loading |
|---|---|
| 16 Word frequency (logarithm) | -0.88 |
| 17 Content word frequency (logarithm) | -0.47 |
| 40 Lexical diversity—all words | +0.42 |
| 18 Minimum word frequency per sentence | -0.37 |

## Marker scores: every name on every Varimax component

Sign-adjusted mean marker loading before orienting the components; the assignment uses the absolute values. Bold: assigned component.

| Name | Varimax 1 | Varimax 2 | Varimax 3 | Varimax 4 | Varimax 5 | Varimax 6 | Varimax 7 | Varimax 8 |
|---|---|---|---|---|---|---|---|---|
| Narrativity | **-0.63** | -0.01 | +0.03 | +0.09 | +0.04 | +0.12 | -0.02 | +0.08 |
| Syntactic simplicity | -0.20 | -0.12 | **-0.42** | +0.03 | -0.06 | -0.05 | -0.08 | +0.31 |
| Word concreteness | -0.05 | -0.04 | -0.11 | **+0.84** | +0.02 | -0.04 | -0.05 | -0.01 |
| Referential cohesion | +0.00 | **+0.83** | -0.06 | -0.03 | +0.02 | +0.01 | +0.02 | -0.11 |
| Deep cohesion | -0.13 | +0.08 | +0.04 | -0.19 | +0.07 | +0.08 | **+0.64** | -0.01 |
| Verb cohesion | -0.27 | +0.10 | -0.27 | +0.01 | +0.05 | +0.17 | -0.04 | **+0.25** |
| Connectivity | -0.16 | +0.02 | +0.07 | -0.05 | **+0.68** | -0.01 | +0.09 | -0.07 |
| Temporality | -0.08 | +0.09 | -0.15 | +0.05 | +0.14 | **-0.00** | +0.01 | +0.02 |

## Eigenvalues (unrotated PCA)

Components with eigenvalue > 1 (Kaiser criterion): 12; 8 are kept as in the original.

| Component | Eigenvalue | Explained variance | Cumulative |
|---|---|---|---|
| 1 | 7.83 | 16.3% | 16.3% |
| 2 | 5.40 | 11.2% | 27.6% |
| 3 | 5.30 | 11.0% | 38.6% |
| 4 | 3.23 | 6.7% | 45.3% |
| 5 | 2.34 | 4.9% | 50.2% |
| 6 | 1.84 | 3.8% | 54.0% |
| 7 | 1.60 | 3.3% | 57.3% |
| 8 | 1.47 | 3.1% | 60.4% |
| 9 | 1.36 | 2.8% | 63.2% |
| 10 | 1.27 | 2.7% | 65.9% |
| 11 | 1.12 | 2.3% | 68.2% |
| 12 | 1.03 | 2.2% | 70.4% |
| 13 | 0.97 | 2.0% | 72.4% |
| 14 | 0.96 | 2.0% | 74.4% |
| 15 | 0.89 | 1.9% | 76.3% |

## Input measure pairs with |r| > 0.9

Removed by Graesser et al. (2011) before their PCA; kept here (see OPEN_QUESTIONS.md).

| Measure | Measure | r |
|---|---|---|
| 13 Logical connectives | 14 Additive connectives | +0.91 |
| 21 Concreteness | 22 Imagery | +0.92 |
| 36 Argument overlap—all sentences | 38 Stem overlap—all sentences | +0.99 |
| 43 LSA overlap—adjacent sentences | 44 LSA overlap—all sentences | +0.91 |
