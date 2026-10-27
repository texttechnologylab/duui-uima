# Text Easability PCA: CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, translated into German with Kolibri-1 Q4_K_M (translation_kolibri-1-q4km), 4669 of 4724 excerpts (without truncated translations and paragraph mismatches)

Created 2026-10-27T10:00:56+00:00 by `easability/pca.py` from `clear_corpus/indices_de_kolibri-1-q4km.csv`; model `src/main/resources/easability/text_easability_de.json`. Method: Graesser, McNamara & Kulikowich (2011), see the docstring of `pca.py`.

## How to read this report

- **Loading**: correlation of an input measure with a component (−1 to +1), oriented so that a higher component score means easier text. Loadings with |l| ≥ .40 are usually read as relevant.
- **Explained variance**: share of the variance of all input measures that a component accounts for; compare with the published share (Graesser et al. 2011, TASA).
- **Marker score**: mean loading of a name's marker measures (`components.csv`, sign-adjusted so that positive = easier). About 1 = the markers load strongly on this component, about 0 = not at all. Every name is assigned to exactly one component (maximal sum of |marker score|), so a name is assigned even if no component fits it; a low marker score or a higher score on another component means the name does not fit.
- **r Lexile**: Spearman correlation of the component score with the Lexile band (higher = harder), comparable with the published correlation with DRP grade level (higher = harder). A negative value means the component is higher (easier) for easier texts. **r BT** (teacher-rated easiness, higher = easier) is expected with the opposite sign.
- **Literary / informational**: mean z score per CLEAR genre (the reference corpus has mean 0, SD 1).

## Data

- Documents: 4669, of which 954 with at least one missing input value (replaced by the corpus mean).
- Input measures: 48 of the 53 published measures.
- Not available in Coh-Metrix 3.0: 9 Ratio of function words to content words, 41 Lexical diversity—verbs, 52 Verb tense repetition, 53 Verb aspect repetition.
- Excluded: 48 Intentional cohesion (intentional verbs probably undercounted, undefined for 71% of CLEAR (see OPEN_QUESTIONS.md)).

| Measure | Missing documents | Share |
|---|---|---|
| 19 Age of acquisition | 1 | 0.0% |
| 35 Argument overlap—adjacent sentences | 1 | 0.0% |
| 44 LSA overlap—all sentences | 8 | 0.2% |
| 47 Causal cohesion | 947 | 20.3% |
| 49 Verb overlap—adjacent | 2 | 0.0% |
| 51 Temporal cohesion | 1 | 0.0% |

## Components compared with the original

Total explained variance of the 8 components: 58.5% (published: 67.3%).

| Component | Varimax component | Marker score | Higher marker score elsewhere | Variance | Published variance | r Lexile | Published r DRP | r BT | Literary | Informational |
|---|---|---|---|---|---|---|---|---|---|---|
| PCNARz Narrativity | 1 | 0.61 | – | 13.9% | 18.5% | -0.53 | -0.69 | +0.38 | +0.62 | -0.64 |
| PCSYNz Syntactic simplicity | 7 | 0.20 | varimax 3: 0.43 | 6.4% | 9.5% | -0.49 | -0.47 | +0.24 | +0.03 | -0.04 |
| PCCNCz Word concreteness | 4 | 0.84 | – | 6.4% | 6.3% | -0.18 | -0.23 | +0.24 | +0.29 | -0.31 |
| PCREFz Referential cohesion | 2 | 0.80 | – | 10.2% | 14.1% | +0.18 | +0.03 | +0.10 | -0.12 | +0.13 |
| PCDCz Deep cohesion | 8 | 0.44 | – | 3.8% | 5.5% | -0.03 | +0.11 | +0.07 | -0.00 | +0.00 |
| PCVERBz Verb cohesion | 3 | 0.50 | – | 8.4% | 5.4% | -0.35 | – | +0.21 | -0.11 | +0.11 |
| PCCONNz Connectivity | 5 | 0.69 | – | 5.7% | 4.1% | +0.08 | – | +0.02 | +0.12 | -0.13 |
| PCTEMPz Temporality | 6 | 0.07 | varimax 8: 0.12 | 3.7% | 4.0% | +0.01 | – | +0.04 | +0.03 | -0.03 |

## Strongest loadings per component

**PCNARz Narrativity** (Varimax component 1; markers: 6:+;8:+;16:+;3:+;2:-;1:-;19:-)

| Measure | Loading |
|---|---|
| 6 Pronouns | +0.89 |
| 2 Nouns | -0.77 |
| 1 Syllables per word | -0.75 |
| 8 Third-person pronouns | +0.75 |
| 3 Verbs | +0.73 |
| 27 Polysemy—multiple senses of word | +0.71 |
| 29 Modifiers per noun phrase | -0.70 |
| 4 Adjectives | -0.61 |
| 7 First-person pronouns | +0.45 |
| 26 Intentional actions, events, and particles | +0.45 |
| 31 Passive constructions | -0.44 |
| 5 Adverbs | +0.43 |
| 19 Age of acquisition | -0.42 |
| 24 Negations | +0.41 |
| 37 Noun overlap—adjacent sentences | -0.40 |
| 42 LSA-givenness versus newness | +0.38 |
| 25 Causal verbs | +0.32 |

**PCSYNz Syntactic simplicity** (Varimax component 7; markers: 28:-;29:-;30:-;45:-;32:+)

| Measure | Loading |
|---|---|
| 43 LSA overlap—adjacent sentences | -0.79 |
| 44 LSA overlap—all sentences | -0.78 |
| 28 Words per sentence | -0.63 |
| 18 Minimum word frequency per sentence | +0.49 |
| 45 Dissimilarity of parts of speech between sentences | +0.44 |
| 39 Type-token ratio | -0.40 |
| 30 Words before main verb of main clause | -0.39 |

**PCCNCz Word concreteness** (Varimax component 4; markers: 21:+;22:+;20:+)

| Measure | Loading |
|---|---|
| 22 Imagery | +0.93 |
| 21 Concreteness | +0.90 |
| 20 Meaningfulness | +0.69 |
| 19 Age of acquisition | -0.61 |
| 1 Syllables per word | -0.35 |
| 17 Content word frequency (logarithm) | -0.32 |

**PCREFz Referential cohesion** (Varimax component 2; markers: 33:+;34:+;35:+;36:+;37:+;38:+)

| Measure | Loading |
|---|---|
| 38 Stem overlap—all sentences | +0.89 |
| 36 Argument overlap—all sentences | +0.89 |
| 35 Argument overlap—adjacent sentences | +0.85 |
| 34 Content word overlap—all sentences | +0.80 |
| 33 Content word overlap—adjacent sentences | +0.70 |
| 37 Noun overlap—adjacent sentences | +0.65 |
| 40 Lexical diversity—all words | -0.45 |
| 43 LSA overlap—adjacent sentences | +0.40 |
| 44 LSA overlap—all sentences | +0.39 |
| 39 Type-token ratio | -0.37 |
| 28 Words per sentence | +0.35 |

**PCDCz Deep cohesion** (Varimax component 8; markers: 11:+;47:+)

| Measure | Loading |
|---|---|
| 5 Adverbs | +0.50 |
| 11 Causal connectives | +0.45 |
| 47 Causal cohesion | +0.42 |
| 23 Familiarity | +0.41 |
| 12 Temporal connectives | +0.41 |
| 10 Connectives | +0.37 |
| 15 Adversative connectives | +0.32 |

**PCVERBz Verb cohesion** (Varimax component 3; markers: 49:+;50:+)

| Measure | Loading |
|---|---|
| 46 Dissimilarity of words between sentences | -0.72 |
| 32 Syntactic similarity—sentences in paragraph | +0.68 |
| 42 LSA-givenness versus newness | +0.67 |
| 39 Type-token ratio | -0.64 |
| 45 Dissimilarity of parts of speech between sentences | -0.62 |
| 40 Lexical diversity—all words | -0.58 |
| 50 LSA verb overlap—adjacent sentences | +0.50 |
| 49 Verb overlap—adjacent | +0.49 |
| 28 Words per sentence | -0.44 |
| 33 Content word overlap—adjacent sentences | +0.43 |

**PCCONNz Connectivity** (Varimax component 5; markers: 13:+;14:+;15:+;10:+)

| Measure | Loading |
|---|---|
| 13 Logical connectives | +0.94 |
| 14 Additive connectives | +0.93 |
| 10 Connectives | +0.86 |

**PCTEMPz Temporality** (Varimax component 6; markers: 51:+;12:+)

| Measure | Loading |
|---|---|
| 16 Word frequency (logarithm) | +0.81 |
| 17 Content word frequency (logarithm) | +0.54 |
| 7 First-person pronouns | -0.46 |
| 18 Minimum word frequency per sentence | +0.32 |

## Marker scores: every name on every Varimax component

Sign-adjusted mean marker loading before orienting the components; the assignment uses the absolute values. Bold: assigned component.

| Name | Varimax 1 | Varimax 2 | Varimax 3 | Varimax 4 | Varimax 5 | Varimax 6 | Varimax 7 | Varimax 8 |
|---|---|---|---|---|---|---|---|---|
| Narrativity | **+0.61** | +0.04 | +0.01 | +0.15 | +0.03 | +0.16 | -0.01 | +0.04 |
| Syntactic simplicity | +0.24 | -0.11 | -0.43 | +0.02 | -0.04 | -0.10 | **+0.20** | -0.06 |
| Word concreteness | +0.05 | -0.01 | -0.06 | **+0.84** | +0.01 | -0.02 | +0.03 | -0.08 |
| Referential cohesion | +0.05 | **+0.80** | -0.16 | -0.01 | -0.00 | +0.01 | -0.15 | +0.04 |
| Deep cohesion | -0.03 | -0.02 | -0.06 | +0.00 | +0.06 | -0.07 | -0.08 | **+0.44** |
| Verb cohesion | -0.12 | +0.11 | **-0.50** | -0.00 | +0.03 | +0.02 | +0.09 | +0.08 |
| Connectivity | +0.08 | +0.03 | +0.05 | -0.01 | **+0.69** | +0.04 | -0.02 | +0.13 |
| Temporality | +0.01 | +0.05 | -0.08 | +0.06 | +0.04 | **+0.07** | -0.08 | +0.12 |

## Eigenvalues (unrotated PCA)

Components with eigenvalue > 1 (Kaiser criterion): 13; 8 are kept as in the original.

| Component | Eigenvalue | Explained variance | Cumulative |
|---|---|---|---|
| 1 | 7.59 | 15.8% | 15.8% |
| 2 | 6.02 | 12.5% | 28.4% |
| 3 | 4.41 | 9.2% | 37.5% |
| 4 | 3.02 | 6.3% | 43.8% |
| 5 | 2.36 | 4.9% | 48.8% |
| 6 | 1.69 | 3.5% | 52.3% |
| 7 | 1.53 | 3.2% | 55.5% |
| 8 | 1.45 | 3.0% | 58.5% |
| 9 | 1.31 | 2.7% | 61.2% |
| 10 | 1.14 | 2.4% | 63.6% |
| 11 | 1.12 | 2.3% | 65.9% |
| 12 | 1.06 | 2.2% | 68.1% |
| 13 | 1.02 | 2.1% | 70.2% |
| 14 | 0.95 | 2.0% | 72.2% |
| 15 | 0.92 | 1.9% | 74.2% |

## Input measure pairs with |r| > 0.9

Removed by Graesser et al. (2011) before their PCA; kept here (see OPEN_QUESTIONS.md).

| Measure | Measure | r |
|---|---|---|
| 21 Concreteness | 22 Imagery | +0.92 |
| 36 Argument overlap—all sentences | 38 Stem overlap—all sentences | +1.00 |
