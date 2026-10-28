# Text Easability components: EN vs. DE

Created 2026-10-27T10:00:57+00:00 by `easability/compare_models.py` from `easability/results/clear_en_pca` (EN) and `easability/results/clear_de_kolibri-1-q4km_pca` (DE).

## How to read

- **Congruence** (Tucker's phi): similarity of the loadings of the component in EN and DE over 48 shared input measures; >= 0.95 equal, 0.85-0.94 fairly similar, below 0.85 different (Lorenzo-Seva & ten Berge 2006). If another DE component has a higher congruence, the components are swapped or mixed.
- **r texts**: correlation of the component scores of the same 4669 texts in EN and DE. High means the component ranks the texts the same way in both languages.

## Components

| Component | Name | Congruence | Rating | Best match in DE | r texts (Pearson) | Spearman |
|---|---|---|---|---|---|---|
| PCNARz | Narrativity | 0.88 | fairly similar |  | 0.90 | 0.91 |
| PCSYNz | Syntactic simplicity | 0.05 | different | PCVERBz (0.93) | -0.04 | -0.05 |
| PCCNCz | Word concreteness | 0.94 | fairly similar |  | 0.71 | 0.70 |
| PCREFz | Referential cohesion | 0.97 | equal |  | 0.71 | 0.70 |
| PCDCz | Deep cohesion | 0.64 | different |  | 0.30 | 0.29 |
| PCVERBz | Verb cohesion | 0.47 | different | PCSYNz (0.90) | 0.35 | 0.35 |
| PCCONNz | Connectivity | 0.97 | equal |  | 0.87 | 0.86 |
| PCTEMPz | Temporality | -0.79 | different |  | -0.51 | -0.50 |

## Congruence matrix (rows EN, columns DE)

|  | PCNARz | PCSYNz | PCCNCz | PCREFz | PCDCz | PCVERBz | PCCONNz | PCTEMPz |
|---|---|---|---|---|---|---|---|---|
| PCNARz | 0.88 | 0.24 | 0.17 | -0.00 | 0.40 | 0.05 | 0.10 | 0.06 |
| PCSYNz | 0.02 | 0.05 | 0.21 | 0.15 | -0.16 | 0.93 | -0.14 | -0.04 |
| PCCNCz | 0.06 | 0.10 | 0.94 | -0.05 | -0.24 | 0.18 | 0.01 | 0.01 |
| PCREFz | -0.00 | -0.37 | -0.03 | 0.97 | 0.14 | 0.30 | 0.05 | 0.08 |
| PCDCz | -0.15 | -0.11 | -0.14 | 0.04 | 0.64 | -0.20 | 0.12 | 0.11 |
| PCVERBz | 0.38 | 0.90 | 0.16 | -0.29 | 0.10 | 0.47 | -0.17 | -0.05 |
| PCCONNz | 0.02 | -0.14 | 0.06 | 0.05 | 0.11 | -0.15 | 0.97 | 0.16 |
| PCTEMPz | -0.08 | 0.04 | 0.05 | -0.18 | -0.08 | -0.16 | -0.09 | -0.79 |
