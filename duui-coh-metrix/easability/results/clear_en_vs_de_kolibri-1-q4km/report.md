# Text Easability components: EN vs. DE

Created 2026-11-04T21:21:20+00:00 by `easability/compare_models.py` from `easability/results/clear_en_pca` (EN) and `easability/results/clear_de_kolibri-1-q4km_pca` (DE).

## How to read

- **Congruence** (Tucker's phi): similarity of the loadings of the component in EN and DE over 49 shared input measures; >= 0.95 equal, 0.85-0.94 fairly similar, below 0.85 different (Lorenzo-Seva & ten Berge 2006). If another DE component has a higher congruence, the components are swapped or mixed.
- **r texts**: correlation of the component scores of the same 4669 texts in EN and DE. High means the component ranks the texts the same way in both languages.

## Components

| Component | Name | Congruence | Rating | Best match in DE | r texts (Pearson) | Spearman |
|---|---|---|---|---|---|---|
| PCNARz | Narrativity | 0.90 | fairly similar |  | 0.89 | 0.90 |
| PCSYNz | Syntactic simplicity | 0.89 | fairly similar |  | 0.67 | 0.65 |
| PCCNCz | Word concreteness | 0.94 | fairly similar |  | 0.83 | 0.83 |
| PCREFz | Referential cohesion | 0.98 | equal |  | 0.78 | 0.76 |
| PCDCz | Deep cohesion | 0.97 | equal |  | 0.48 | 0.48 |
| PCVERBz | Verb cohesion | 0.48 | different | PCNARz (0.53) | 0.13 | 0.07 |
| PCCONNz | Connectivity | 0.96 | equal |  | 0.82 | 0.80 |
| PCTEMPz | Temporality | 0.19 | different | PCVERBz (0.93) | 0.14 | 0.19 |

## Congruence matrix (rows EN, columns DE)

|  | PCNARz | PCSYNz | PCCNCz | PCREFz | PCDCz | PCVERBz | PCCONNz | PCTEMPz |
|---|---|---|---|---|---|---|---|---|
| PCNARz | 0.90 | 0.12 | 0.17 | 0.01 | 0.12 | 0.02 | 0.39 | -0.31 |
| PCSYNz | 0.23 | 0.89 | 0.26 | -0.26 | -0.17 | 0.27 | -0.06 | 0.10 |
| PCCNCz | 0.28 | 0.29 | 0.94 | -0.05 | -0.07 | 0.10 | 0.19 | -0.12 |
| PCREFz | -0.02 | -0.31 | -0.07 | 0.98 | 0.11 | 0.22 | 0.09 | 0.05 |
| PCDCz | 0.12 | -0.08 | -0.09 | 0.13 | 0.97 | -0.01 | 0.15 | -0.17 |
| PCVERBz | 0.53 | 0.49 | 0.28 | 0.20 | -0.02 | 0.48 | 0.25 | -0.44 |
| PCCONNz | 0.63 | -0.01 | 0.30 | 0.13 | 0.14 | -0.02 | 0.96 | -0.07 |
| PCTEMPz | 0.03 | 0.43 | 0.19 | 0.23 | -0.09 | 0.93 | -0.10 | 0.19 |
