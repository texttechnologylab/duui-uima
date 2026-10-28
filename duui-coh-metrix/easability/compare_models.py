"""Compare two Text Easability PCAs of the same texts (e.g. the English CLEAR corpus and its German translation).

- Loadings: Tucker's congruence coefficient between the components (phi = sum(a * b) / sqrt(sum(a^2) * sum(b^2))
  over the input measures of both PCAs). Rule of thumb (Lorenzo-Seva & ten Berge 2006): >= .95 equal,
  .85-.94 fairly similar, below .85 different. The diagonal compares the components of the same name; the full
  matrix shows swapped components.
- Texts: Pearson and Spearman correlation of the component scores of the same documents (document_id).

Writes to <output_dir>: congruence.csv (matrix, rows: components of A, columns: components of B),
score_correlations.csv and report.md.

Usage: python compare_models.py <pca_dir_a> <pca_dir_b> <output_dir> --name-a EN --name-b DE
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

EQUAL = 0.95
SIMILAR = 0.85


def congruence(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Tucker's phi between all columns of a and b (rows: the same measures)."""
    numerator = a.T.values @ b.values
    denominator = np.sqrt(np.outer((a.values ** 2).sum(axis=0), (b.values ** 2).sum(axis=0)))
    return pd.DataFrame(numerator / denominator, index=a.columns, columns=b.columns)


def rating(phi: float) -> str:
    phi = abs(phi)
    return "equal" if phi >= EQUAL else "fairly similar" if phi >= SIMILAR else "different"


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pca_a", type=Path, help="output directory of pca.py")
    parser.add_argument("pca_b", type=Path, help="output directory of pca.py")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--name-a", required=True)
    parser.add_argument("--name-b", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loadings_a = pd.read_csv(args.pca_a / "loadings.csv", index_col="measure")
    loadings_b = pd.read_csv(args.pca_b / "loadings.csv", index_col="measure")
    measures = loadings_a.index.intersection(loadings_b.index)
    phi = congruence(loadings_a.loc[measures], loadings_b.loc[measures])
    phi.to_csv(args.output_dir / "congruence.csv", index_label="component")

    scores_a = pd.read_csv(args.pca_a / "scores.csv", dtype={"document_id": str}).set_index("document_id")
    scores_b = pd.read_csv(args.pca_b / "scores.csv", dtype={"document_id": str}).set_index("document_id")
    documents = scores_a.index.intersection(scores_b.index)
    names = pd.read_csv(args.pca_a / "assignment.csv").set_index("component")["name"]

    rows = []
    for component in loadings_a.columns:
        pearson = pearsonr(scores_a.loc[documents, component], scores_b.loc[documents, component])
        spearman = spearmanr(scores_a.loc[documents, component], scores_b.loc[documents, component])
        best = phi.loc[component].abs().idxmax()
        rows.append({
            "component": component, "name": names[component],
            "congruence": phi.loc[component, component], "rating": rating(phi.loc[component, component]),
            "best_match_b": best, "best_congruence_b": phi.loc[component, best],
            "pearson": pearson.statistic, "spearman": spearman.statistic,
        })
    correlations = pd.DataFrame(rows)
    correlations.to_csv(args.output_dir / "score_correlations.csv", index=False)

    a, b = args.name_a, args.name_b
    report = [
        f"# Text Easability components: {a} vs. {b}",
        "",
        f"Created {datetime.now(timezone.utc).isoformat(timespec='seconds')} by `easability/compare_models.py` "
        f"from `{args.pca_a}` ({a}) and `{args.pca_b}` ({b}).",
        "",
        "## How to read",
        "",
        f"- **Congruence** (Tucker's phi): similarity of the loadings of the component in {a} and {b} over "
        f"{len(measures)} shared input measures; >= {EQUAL} equal, {SIMILAR}-{EQUAL - 0.01:.2f} fairly similar, "
        f"below {SIMILAR} different (Lorenzo-Seva & ten Berge 2006). If another {b} component has a higher "
        "congruence, the components are swapped or mixed.",
        f"- **r texts**: correlation of the component scores of the same {len(documents)} texts in {a} and {b}. "
        "High means the component ranks the texts the same way in both languages.",
        "",
        "## Components",
        "",
        table(["Component", "Name", "Congruence", "Rating", f"Best match in {b}", "r texts (Pearson)",
               "Spearman"],
              [[r.component, r.name, f"{r.congruence:.2f}", r.rating,
                "" if r.best_match_b == r.component else f"{r.best_match_b} ({r.best_congruence_b:.2f})",
                f"{r.pearson:.2f}", f"{r.spearman:.2f}"] for r in correlations.itertuples()]),
        "",
        f"## Congruence matrix (rows {a}, columns {b})",
        "",
        table([""] + list(phi.columns),
              [[component] + [f"{value:.2f}" for value in phi.loc[component]] for component in phi.index]),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"{len(measures)} measures, {len(documents)} shared documents; report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
