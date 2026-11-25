# Text Easability components: EN vs. DE

Created 2026-11-19T15:05:15+00:00 by `easability/export_duui_results.py` of coh-metrix_resurrection, with the method of duui-coh-metrix's `compare_models.py`: loadings from `refa_en_pca` (EN) and `refv1_de_pca` (DE); scores of CLEAR's test half, the English excerpts scored with the English model and their German translations with the German model. The two reference corpora share no texts, so CLEAR stands in for them here (in `clear_en_vs_de_kolibri-1-q4km` the reference corpus itself is shared).

## How to read

- **Congruence** (Tucker's phi): similarity of the loadings of the component in EN and DE over 44 shared input measures; >= 0.95 equal, 0.85-0.94 fairly similar, below 0.85 different (Lorenzo-Seva & ten Berge 2006). If another DE component has a higher congruence, the components are swapped or mixed. The English components are named by this congruence (one-to-one, summed |phi| maximal), so the diagonal holds the matches.
- **r texts**: correlation of the component scores of the same 2359 CLEAR texts in EN and DE. High means the component ranks the texts the same way in both languages.

## Components

| Component | Name | Congruence | Rating | Best match in DE | r texts (Pearson) | Spearman |
|---|---|---|---|---|---|---|
| PCNARz | Narrativity | 0.87 | fairly similar |  | 0.82 | 0.81 |
| PCSYNz | Syntactic simplicity | 0.81 | different |  | 0.66 | 0.66 |
| PCCNCz | Word concreteness | 0.88 | fairly similar |  | 0.81 | 0.81 |
| PCREFz | Referential cohesion | 0.95 | fairly similar |  | 0.79 | 0.78 |
| PCDCz | Deep cohesion | 0.96 | equal |  | 0.45 | 0.42 |
| PCVERBz | Semantic overlap | 0.73 | different |  | 0.63 | 0.54 |
| PCCONNz | Connectivity | 0.85 | fairly similar |  | 0.74 | 0.72 |
| PCTEMPz | Third-person narration | 0.43 | different |  | 0.56 | 0.51 |

## Congruence matrix (rows EN, columns DE)

|  | PCNARz | PCSYNz | PCCNCz | PCREFz | PCDCz | PCVERBz | PCCONNz | PCTEMPz |
|---|---|---|---|---|---|---|---|---|
| PCNARz | 0.87 | 0.45 | 0.19 | 0.03 | -0.02 | -0.06 | 0.54 | 0.41 |
| PCSYNz | 0.18 | 0.81 | 0.32 | -0.12 | -0.12 | -0.35 | -0.09 | 0.07 |
| PCCNCz | -0.06 | 0.12 | 0.88 | -0.07 | -0.21 | -0.11 | -0.28 | 0.04 |
| PCREFz | -0.16 | -0.33 | -0.14 | 0.95 | 0.06 | 0.34 | -0.07 | 0.01 |
| PCDCz | 0.14 | -0.02 | -0.19 | 0.04 | 0.96 | 0.01 | 0.34 | -0.05 |
| PCVERBz | -0.04 | 0.55 | 0.09 | 0.30 | -0.03 | 0.73 | -0.04 | -0.05 |
| PCCONNz | 0.60 | 0.32 | 0.25 | 0.11 | 0.02 | -0.02 | 0.85 | 0.55 |
| PCTEMPz | -0.07 | -0.24 | -0.07 | -0.27 | 0.13 | -0.27 | -0.28 | 0.43 |
