# Text Easability PCA: CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, translated into German with Kolibri-1 Q4_K_M (translation_kolibri-1-q4km), excerpts without truncated translations and paragraph mismatches; duui-coh-metrix 0.6.0, default pipeline

Created 2026-11-04T21:21:19+00:00 by `easability/pca.py` from `/storage/projects/CORE/projects2/coh_metrix/clear_corpus/indices_de_kolibri-1-q4km_0.6.0.csv`; model `src/main/resources/easability/text_easability_de_cleartr.json`. Method: Graesser, McNamara & Kulikowich (2011), see the docstring of `pca.py`.

## How to read this report

- **Loading**: correlation of an input measure with a component (−1 to +1), oriented so that a higher component score means easier text. Loadings with |l| ≥ .40 are usually read as relevant.
- **Explained variance**: share of the variance of all input measures that a component accounts for; compare with the published share (Graesser et al. 2011, TASA).
- **Marker score**: mean loading of a name's marker measures (`components.csv`, sign-adjusted so that positive = easier). About 1 = the markers load strongly on this component, about 0 = not at all. Every name is assigned to exactly one component (maximal sum of |marker score|), so a name is assigned even if no component fits it; a low marker score or a higher score on another component means the name does not fit.
- **r Lexile**: Spearman correlation of the component score with the Lexile band (higher = harder), comparable with the published correlation with DRP grade level (higher = harder). A negative value means the component is higher (easier) for easier texts. **r BT** (teacher-rated easiness, higher = easier) is expected with the opposite sign.
- **Literary / informational**: mean z score per CLEAR genre (the reference corpus has mean 0, SD 1).

## Data

- Documents: 4669, of which 32 with at least one missing input value (replaced by the corpus mean).
- Input measures: 49 of the 53 published measures.
- Not available in Coh-Metrix 3.0: 9 Ratio of function words to content words, 41 Lexical diversity—verbs, 52 Verb tense repetition, 53 Verb aspect repetition.
- Excluded: .

| Measure | Missing documents | Share |
|---|---|---|
| 32 Syntactic similarity—sentences in paragraph | 2 | 0.0% |
| 33 Content word overlap—adjacent sentences | 2 | 0.0% |
| 34 Content word overlap—all sentences | 2 | 0.0% |
| 35 Argument overlap—adjacent sentences | 2 | 0.0% |
| 36 Argument overlap—all sentences | 2 | 0.0% |
| 37 Noun overlap—adjacent sentences | 2 | 0.0% |
| 38 Stem overlap—all sentences | 2 | 0.0% |
| 43 LSA overlap—adjacent sentences | 2 | 0.0% |
| 44 LSA overlap—all sentences | 31 | 0.7% |
| 45 Dissimilarity of parts of speech between sentences | 2 | 0.0% |
| 46 Dissimilarity of words between sentences | 2 | 0.0% |
| 49 Verb overlap—adjacent | 1 | 0.0% |
| 50 LSA verb overlap—adjacent sentences | 1 | 0.0% |
| 51 Temporal cohesion | 3 | 0.1% |

## Components compared with the original

Total explained variance of the 8 components: 62.6% (published: 67.3%).

| Component | Varimax component | Marker score | Higher marker score elsewhere | Variance | Published variance | r Lexile | Published r DRP | r BT | Literary | Informational |
|---|---|---|---|---|---|---|---|---|---|---|
| PCNARz Narrativity | 1 | 0.70 | – | 16.9% | 18.5% | -0.49 | -0.69 | +0.34 | +0.52 | -0.54 |
| PCSYNz Syntactic simplicity | 7 | 0.32 | varimax 3: 0.38 | 8.5% | 9.5% | -0.57 | -0.47 | +0.31 | +0.01 | -0.01 |
| PCCNCz Word concreteness | 4 | 0.74 | – | 6.7% | 6.3% | -0.24 | -0.23 | +0.30 | +0.43 | -0.45 |
| PCREFz Referential cohesion | 2 | 0.80 | – | 10.2% | 14.1% | +0.10 | +0.03 | +0.11 | -0.10 | +0.10 |
| PCDCz Deep cohesion | 6 | 0.89 | – | 4.0% | 5.5% | +0.03 | +0.11 | -0.01 | +0.02 | -0.02 |
| PCVERBz Verb cohesion | 3 | 0.38 | – | 6.2% | 5.4% | -0.17 | – | +0.15 | -0.06 | +0.07 |
| PCCONNz Connectivity | 5 | 0.67 | – | 7.1% | 4.1% | -0.05 | – | +0.14 | +0.27 | -0.28 |
| PCTEMPz Temporality | 8 | 0.11 | varimax 4: 0.25 | 3.0% | 4.0% | +0.02 | – | -0.03 | -0.08 | +0.08 |

## Strongest loadings per component

**PCNARz Narrativity** (Varimax component 1; markers: 6:+;8:+;16:+;3:+;2:-;1:-;19:-)

| Measure | Loading |
|---|---|
| 6 Pronouns | +0.89 |
| 16 Word frequency (logarithm) | +0.83 |
| 17 Content word frequency (logarithm) | +0.82 |
| 2 Nouns | -0.74 |
| 29 Modifiers per noun phrase | -0.65 |
| 1 Syllables per word | -0.65 |
| 18 Minimum word frequency per sentence | +0.64 |
| 3 Verbs | +0.62 |
| 27 Polysemy—multiple senses of word | +0.61 |
| 8 Third-person pronouns | +0.60 |
| 7 First-person pronouns | +0.57 |
| 19 Age of acquisition | -0.55 |
| 24 Negations | +0.48 |
| 31 Passive constructions | -0.47 |
| 5 Adverbs | +0.47 |
| 26 Intentional actions, events, and particles | +0.45 |
| 50 LSA verb overlap—adjacent sentences | -0.44 |
| 23 Familiarity | +0.42 |
| 10 Connectives | +0.40 |
| 4 Adjectives | -0.39 |
| 37 Noun overlap—adjacent sentences | -0.32 |
| 34 Content word overlap—all sentences | +0.31 |

**PCSYNz Syntactic simplicity** (Varimax component 7; markers: 28:-;29:-;30:-;45:-;32:+)

| Measure | Loading |
|---|---|
| 43 LSA overlap—adjacent sentences | -0.82 |
| 44 LSA overlap—all sentences | -0.81 |
| 28 Words per sentence | -0.66 |
| 39 Type-token ratio | -0.58 |
| 18 Minimum word frequency per sentence | +0.54 |
| 32 Syntactic similarity—sentences in paragraph | +0.46 |
| 4 Adjectives | -0.40 |
| 38 Stem overlap—all sentences | -0.40 |
| 36 Argument overlap—all sentences | -0.40 |
| 30 Words before main verb of main clause | -0.38 |
| 40 Lexical diversity—all words | -0.37 |
| 49 Verb overlap—adjacent | +0.34 |
| 21 Concreteness | +0.31 |

**PCCNCz Word concreteness** (Varimax component 4; markers: 21:+;22:+;20:+)

| Measure | Loading |
|---|---|
| 22 Imagery | +0.83 |
| 21 Concreteness | +0.79 |
| 19 Age of acquisition | -0.63 |
| 20 Meaningfulness | +0.59 |
| 1 Syllables per word | -0.49 |
| 27 Polysemy—multiple senses of word | +0.33 |
| 12 Temporal connectives | +0.31 |

**PCREFz Referential cohesion** (Varimax component 2; markers: 33:+;34:+;35:+;36:+;37:+;38:+)

| Measure | Loading |
|---|---|
| 38 Stem overlap—all sentences | +0.84 |
| 36 Argument overlap—all sentences | +0.84 |
| 35 Argument overlap—adjacent sentences | +0.83 |
| 34 Content word overlap—all sentences | +0.80 |
| 33 Content word overlap—adjacent sentences | +0.76 |
| 37 Noun overlap—adjacent sentences | +0.74 |
| 40 Lexical diversity—all words | -0.54 |
| 39 Type-token ratio | -0.48 |
| 28 Words per sentence | +0.32 |
| 43 LSA overlap—adjacent sentences | +0.31 |

**PCDCz Deep cohesion** (Varimax component 6; markers: 11:+;47:+)

| Measure | Loading |
|---|---|
| 47 Causal cohesion | +0.94 |
| 11 Causal connectives | +0.84 |
| 25 Causal verbs | -0.37 |

**PCVERBz Verb cohesion** (Varimax component 3; markers: 49:+;50:+)

| Measure | Loading |
|---|---|
| 42 LSA-givenness versus newness | +0.76 |
| 46 Dissimilarity of words between sentences | -0.68 |
| 45 Dissimilarity of parts of speech between sentences | -0.67 |
| 32 Syntactic similarity—sentences in paragraph | +0.58 |
| 39 Type-token ratio | -0.42 |
| 50 LSA verb overlap—adjacent sentences | +0.40 |
| 28 Words per sentence | -0.40 |
| 40 Lexical diversity—all words | -0.39 |
| 49 Verb overlap—adjacent | +0.37 |
| 51 Temporal cohesion | +0.33 |

**PCCONNz Connectivity** (Varimax component 5; markers: 13:+;14:+;15:+;10:+)

| Measure | Loading |
|---|---|
| 13 Logical connectives | +0.88 |
| 14 Additive connectives | +0.87 |
| 10 Connectives | +0.78 |
| 3 Verbs | +0.40 |
| 27 Polysemy—multiple senses of word | +0.35 |
| 2 Nouns | -0.34 |
| 5 Adverbs | +0.32 |
| 12 Temporal connectives | +0.32 |

**PCTEMPz Temporality** (Varimax component 8; markers: 51:+;12:+)

| Measure | Loading |
|---|---|
| 49 Verb overlap—adjacent | -0.41 |
| 5 Adverbs | -0.40 |
| 4 Adjectives | -0.40 |
| 23 Familiarity | -0.37 |
| 15 Adversative connectives | -0.33 |
| 48 Intentional cohesion | +0.33 |

## Marker scores: every name on every Varimax component

Sign-adjusted mean marker loading before orienting the components; the assignment uses the absolute values. Bold: assigned component.

| Name | Varimax 1 | Varimax 2 | Varimax 3 | Varimax 4 | Varimax 5 | Varimax 6 | Varimax 7 | Varimax 8 |
|---|---|---|---|---|---|---|---|---|
| Narrativity | **+0.70** | +0.03 | -0.03 | +0.24 | -0.26 | +0.04 | -0.04 | +0.04 |
| Syntactic simplicity | +0.18 | -0.05 | -0.38 | +0.04 | -0.00 | -0.03 | **-0.32** | -0.13 |
| Word concreteness | +0.12 | -0.01 | -0.07 | **+0.74** | -0.07 | -0.02 | -0.15 | +0.02 |
| Referential cohesion | +0.08 | **+0.80** | -0.07 | -0.01 | -0.05 | +0.04 | +0.16 | -0.02 |
| Deep cohesion | +0.13 | +0.08 | +0.02 | -0.05 | -0.07 | **+0.89** | +0.08 | -0.02 |
| Verb cohesion | -0.24 | +0.16 | **-0.38** | -0.02 | -0.06 | -0.04 | -0.15 | +0.28 |
| Connectivity | +0.26 | +0.07 | +0.04 | +0.02 | **-0.67** | +0.05 | +0.06 | +0.10 |
| Temporality | +0.03 | +0.02 | -0.15 | +0.25 | -0.24 | +0.03 | +0.07 | **-0.11** |

## Eigenvalues (unrotated PCA)

Components with eigenvalue > 1 (Kaiser criterion): 12; 8 are kept as in the original.

| Component | Eigenvalue | Explained variance | Cumulative |
|---|---|---|---|
| 1 | 10.64 | 21.7% | 21.7% |
| 2 | 5.90 | 12.0% | 33.7% |
| 3 | 4.81 | 9.8% | 43.6% |
| 4 | 2.71 | 5.5% | 49.1% |
| 5 | 2.00 | 4.1% | 53.2% |
| 6 | 1.78 | 3.6% | 56.8% |
| 7 | 1.51 | 3.1% | 59.9% |
| 8 | 1.31 | 2.7% | 62.6% |
| 9 | 1.17 | 2.4% | 65.0% |
| 10 | 1.11 | 2.3% | 67.2% |
| 11 | 1.03 | 2.1% | 69.3% |
| 12 | 1.00 | 2.0% | 71.4% |
| 13 | 0.97 | 2.0% | 73.4% |
| 14 | 0.91 | 1.9% | 75.2% |
| 15 | 0.89 | 1.8% | 77.0% |

## Input measure pairs with |r| > 0.9

Removed by Graesser et al. (2011) before their PCA; kept here, the published measures are used unchanged.

| Measure | Measure | r |
|---|---|---|
| 13 Logical connectives | 14 Additive connectives | +0.98 |
| 36 Argument overlap—all sentences | 38 Stem overlap—all sentences | +1.00 |
| 43 LSA overlap—adjacent sentences | 44 LSA overlap—all sentences | +0.92 |
