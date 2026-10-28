"""PCA with Varimax rotation on the Coh-Metrix indices of a reference corpus (Text Easability components).

Follows Graesser, McNamara & Kulikowich (2011): standardized input measures (pca_input_measures.csv, the 53
published measures mapped to our index labels), principal component analysis, orthogonal Varimax rotation,
8 components.

- Missing values are replaced by the measure's mean in the reference corpus (neutral after standardization);
  the counts are reported and written to the model: at runtime only measures that are missing in the reference
  corpus may be missing (e.g. causal cohesion without causal verbs), otherwise the components are undefined
  (e.g. empty documents). Pairs of input measures with |r| > .90 are only reported (Graesser et al. removed
  such pairs before the PCA, but the published 53 measures are used here unchanged).
- Each component is named by its marker measures (components.csv, derived from the component descriptions in
  McNamara et al. 2014, ch. 5): the components are assigned to the names so that the summed marker loadings are
  maximal, and oriented so that higher scores mean easier text.
- Component scores: standardized inputs times the rotated PCA score weights (unit variance in the reference
  corpus, i.e. z scores). Percentiles: share of reference texts with a lower score, from 1001 quantiles.

Writes to <output_dir>: missing_values.csv, high_correlations.csv, eigenvalues.csv, loadings.csv,
explained_variance.csv, assignment.csv, marker_scores.csv, validation.csv, scores.csv, meta.json and report.md
(readable summary); and the model for the DUUI
component (see text_easability in duui.py) to --model.

Usage: python pca.py <indices.csv> <output_dir> --model <model.json> --language en --corpus "<description>"
       --label-suffix _clear
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from factor_analyzer.rotator import Rotator
from scipy.optimize import linear_sum_assignment
from scipy.stats import pointbiserialr, spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

MEASURES_FILE = Path(__file__).parent / "pca_input_measures.csv"
COMPONENTS_FILE = Path(__file__).parent / "components.csv"
HIGH_CORRELATION = 0.90
PERCENTILE_QUANTILES = 1001

# CLEAR difficulty and genre columns (see DUUIClearCorpusReader)
EASINESS_COLUMN = "meta_BT_easiness"
LEXILE_COLUMN = "meta_Lexile Band"
GENRE_COLUMN = "meta_Categ"
GENRE_LITERARY = "Lit"


def load_measures() -> pd.DataFrame:
    """The available and not excluded 2011 measures."""
    measures = pd.read_csv(MEASURES_FILE, dtype=str, keep_default_na=False)
    measures = measures[(measures["labels_ttlab"] != "") & (measures["excluded"] == "")]
    measures["column"] = measures["number"] + " " + measures["measure_2011"]
    return measures.set_index("number")


def input_values(indices: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    """One column per measure; measures with multiple labels are summed (missing if one label is missing)."""
    columns = {}
    for m in measures.itertuples():
        labels = m.labels_ttlab.split(";")
        columns[m.column] = indices[labels].sum(axis=1, min_count=len(labels))
    return pd.DataFrame(columns, index=indices.index)


def marker_scores(loadings: np.ndarray, measures: pd.DataFrame, components: pd.DataFrame) -> np.ndarray:
    """Mean of the direction-signed marker loadings, one row per named component, one column per PC."""
    position = {number: i for i, number in enumerate(measures.index)}
    scores = np.zeros((len(components), loadings.shape[1]))
    for row, markers in enumerate(components["marker_measures"]):
        signed = []
        for marker in markers.split(";"):
            number, direction = marker.split(":")
            if number in position:
                signed.append((1 if direction == "+" else -1) * loadings[position[number]])
        scores[row] = np.mean(signed, axis=0)
    return scores


def lexile(values: pd.Series) -> pd.Series:
    """Lower bound of the Lexile band ('1100', '410L-600L' -> 1100, 410)."""
    return pd.to_numeric(values.astype(str).str.extract(r"^(\d+)")[0], errors="coerce")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("indices", type=Path, help="output of extract_indices.py")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model", type=Path, required=True, help="model file for the DUUI component")
    parser.add_argument("--language", required=True)
    parser.add_argument("--corpus", required=True, help="description of the reference corpus")
    parser.add_argument("--label-suffix", required=True, help="label_ttlab suffix of the components, e.g. _clear")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # low_memory=False: one type per column (CLEAR's Lexile Band mixes numbers and ranges)
    indices = pd.read_csv(args.indices, low_memory=False)
    measures = load_measures()
    components = pd.read_csv(COMPONENTS_FILE, dtype=str, keep_default_na=False)
    n_components = len(components)
    x = input_values(indices, measures)

    missing = x.isna().sum().rename("missing_documents").to_frame()
    missing["missing_share"] = missing["missing_documents"] / len(x)
    missing.to_csv(args.output_dir / "missing_values.csv", index_label="measure")

    corr = x.corr()
    pairs = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
        if abs(corr.loc[a, b]) > HIGH_CORRELATION
    ]
    pd.DataFrame(pairs, columns=["measure_a", "measure_b", "r"]).to_csv(
        args.output_dir / "high_correlations.csv", index=False)

    imputer = SimpleImputer(strategy="mean").fit(x)
    scaler = StandardScaler().fit(imputer.transform(x))
    z = scaler.transform(imputer.transform(x))

    full = PCA().fit(z)
    pd.DataFrame({
        "component": np.arange(1, len(full.explained_variance_) + 1),
        "eigenvalue": full.explained_variance_,
        "explained_variance_ratio": full.explained_variance_ratio_,
    }).to_csv(args.output_dir / "eigenvalues.csv", index=False)

    pca = PCA(n_components=n_components).fit(z)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    rotator = Rotator(method="varimax")
    rotated = rotator.fit_transform(loadings)

    # name and orient the rotated components by their marker measures
    markers = marker_scores(rotated, measures, components)
    rows, columns = linear_sum_assignment(-np.abs(markers))
    order = columns[np.argsort(rows)]
    signs = np.sign(markers[np.arange(n_components), order])
    rotated = rotated[:, order] * signs
    rotation = rotator.rotation_[:, order] * signs
    names = list(components["label_v3_z"])

    # score weights on the standardized inputs: unit-variance PCA scores, rotated like the loadings
    weights = pca.components_.T / np.sqrt(pca.explained_variance_) @ rotation
    scores = pd.DataFrame((z - pca.mean_) @ weights, index=indices.index, columns=names)

    pd.DataFrame(rotated, index=x.columns, columns=names).to_csv(
        args.output_dir / "loadings.csv", index_label="measure")
    pd.DataFrame({
        "component": names,
        "explained_variance_ratio": (rotated ** 2).sum(axis=0) / x.shape[1],
    }).to_csv(args.output_dir / "explained_variance.csv", index=False)
    pd.DataFrame({
        "component": names,
        "name": components["name"],
        "varimax_component": order + 1,
        "marker_score": np.abs(markers[np.arange(n_components), order]),
        "marker_measures": components["marker_measures"],
    }).to_csv(args.output_dir / "assignment.csv", index=False)
    scores.insert(0, "document_id", indices["document_id"])
    scores.to_csv(args.output_dir / "scores.csv", index=False)

    literary = indices[GENRE_COLUMN] == GENRE_LITERARY
    validation = []
    for component, name in zip(names, components["name"]):
        easiness = spearmanr(scores[component], indices[EASINESS_COLUMN])
        lexile_r = spearmanr(scores[component], lexile(indices[LEXILE_COLUMN]), nan_policy="omit")
        genre = pointbiserialr(literary.astype(int), scores[component])
        published = components.loc[components["label_v3_z"] == component, "published_r_drp"].iloc[0]
        validation.append({
            "component": component, "name": name,
            "spearman_lexile": lexile_r.statistic, "p_lexile": lexile_r.pvalue,
            "published_r_drp": float(published) if published else None,
            "spearman_bt_easiness": easiness.statistic, "p_bt_easiness": easiness.pvalue,
            "mean_literary": scores.loc[literary, component].mean(),
            "mean_informational": scores.loc[~literary, component].mean(),
            "pointbiserial_literary": genre.statistic, "p_literary": genre.pvalue,
        })
    pd.DataFrame(validation).to_csv(args.output_dir / "validation.csv", index=False)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    levels = np.linspace(0, 1, PERCENTILE_QUANTILES)
    model = {
        "name": "Text Easability components",
        "language": args.language,
        "reference_corpus": args.corpus,
        "label_suffix": args.label_suffix,
        "documents": len(indices),
        "created_at": created_at,
        "method": "PCA on standardized inputs (missing values: reference mean), Varimax rotation; "
                  "z = standardized inputs @ weights; percentile = share of reference texts with a lower z; "
                  "at runtime only inputs with missing_documents > 0 may be missing (reference mean), "
                  "otherwise the components are undefined",
        "inputs": [
            {"measure_number": int(number), "measure": m.measure_2011, "labels_ttlab": m.labels_ttlab.split(";"),
             "mean": float(mean), "scale": float(scale), "missing_documents": int(missing_documents)}
            for (number, m), mean, scale, missing_documents
            in zip(measures.iterrows(), imputer.statistics_, scaler.scale_, missing["missing_documents"])
        ],
        "components": [
            {"label_v3_z": c.label_v3_z, "label_v3_p": c.label_v3_p, "name": c.name,
             "weights": weights[:, i].tolist(),
             "percentile_quantiles": np.quantile(scores[c.label_v3_z], levels).tolist()}
            for i, c in enumerate(components.itertuples())
        ],
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.model.write_text(json.dumps(model, indent=1))

    meta = {
        "created_at": created_at,
        "indices_file": str(args.indices),
        "model_file": str(args.model),
        "documents": len(indices),
        "documents_with_imputed_values": int(x.isna().any(axis=1).sum()),
        "measures": list(x.columns),
        "measures_published": 53,
        "n_components": n_components,
        "rotation": "varimax (factor_analyzer.rotator.Rotator defaults)",
        "standardization": "sklearn StandardScaler (population SD) after SimpleImputer(strategy='mean')",
        "explained_variance_ratio_total": float((rotated ** 2).sum() / x.shape[1]),
        "high_correlation_threshold": HIGH_CORRELATION,
        "versions": {"python": platform.python_version()} | {
            package: version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn", "factor_analyzer")
        },
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    marker_matrix = pd.DataFrame(markers, index=components["name"],
                                 columns=[f"varimax_{i}" for i in range(1, n_components + 1)])
    marker_matrix.to_csv(args.output_dir / "marker_scores.csv", index_label="name")

    report = write_report(args, meta, measures, components, missing, pairs, full, rotated, x.columns, order,
                          marker_matrix, pd.DataFrame(validation))
    (args.output_dir / "report.md").write_text(report)
    print(f"{len(indices)} documents, {x.shape[1]} measures, {meta['explained_variance_ratio_total']:.1%} "
          f"explained by {n_components} components; report: {args.output_dir / 'report.md'}")


def table(header: list, rows: list) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def number(value, digits: int = 2) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:+.{digits}f}"


def write_report(args, meta, measures, components, missing, pairs, full, rotated, measure_columns, order,
                 marker_matrix, validation) -> str:
    """Readable summary of the run: data, components vs. the published values, assignment, eigenvalues,
    loadings, validation."""
    n_components = len(components)
    all_measures = pd.read_csv(MEASURES_FILE, dtype=str, keep_default_na=False)
    unavailable = all_measures[all_measures["labels_ttlab"] == ""]
    excluded = all_measures[(all_measures["labels_ttlab"] != "") & (all_measures["excluded"] != "")]
    variance = (rotated ** 2).sum(axis=0) / len(measure_columns)
    r = []

    r.append(f"# Text Easability PCA: {args.corpus}\n")
    r.append(f"Created {meta['created_at']} by `easability/pca.py` from `{args.indices}`; model "
             f"`{args.model}`. Method: Graesser, McNamara & Kulikowich (2011), see the docstring of `pca.py`.\n")

    r.append("## How to read this report\n")
    r.append("- **Loading**: correlation of an input measure with a component (−1 to +1), oriented so that a higher "
             "component score means easier text. Loadings with |l| ≥ .40 are usually read as relevant.\n"
             "- **Explained variance**: share of the variance of all input measures that a component accounts for; "
             "compare with the published share (Graesser et al. 2011, TASA).\n"
             "- **Marker score**: mean loading of a name's marker measures (`components.csv`, sign-adjusted so "
             "that positive = easier). About 1 = the markers load strongly on this component, about 0 = not at all. "
             "Every name is assigned to exactly one component (maximal sum of |marker score|), so a name is "
             "assigned even if no component fits it; a low marker score or a higher score on another component "
             "means the name does not fit.\n"
             "- **r Lexile**: Spearman correlation of the component score with the Lexile band (higher = harder), "
             "comparable with the published correlation with DRP grade level (higher = harder). A negative value "
             "means the component is higher (easier) for easier texts. **r BT** (teacher-rated easiness, higher = "
             "easier) is expected with the opposite sign.\n"
             "- **Literary / informational**: mean z score per CLEAR genre (the reference corpus has mean 0, SD 1).\n")

    r.append("## Data\n")
    r.append(f"- Documents: {meta['documents']}, of which {meta['documents_with_imputed_values']} with at least "
             f"one missing input value (replaced by the corpus mean).\n"
             f"- Input measures: {len(measure_columns)} of the 53 published measures.\n"
             f"- Not available in Coh-Metrix 3.0: "
             + ", ".join(f"{m.number} {m.measure_2011}" for m in unavailable.itertuples()) + ".\n"
             f"- Excluded: " + ", ".join(f"{m.number} {m.measure_2011} ({m.excluded})"
                                         for m in excluded.itertuples()) + ".\n")
    r.append(table(["Measure", "Missing documents", "Share"],
                   [[m, int(row.missing_documents), f"{row.missing_share:.1%}"]
                    for m, row in missing[missing["missing_documents"] > 0].iterrows()]) + "\n")

    r.append("## Components compared with the original\n")
    r.append(f"Total explained variance of the {n_components} components: "
             f"{meta['explained_variance_ratio_total']:.1%} (published: 67.3%).\n")
    rows = []
    for i, c in enumerate(components.itertuples()):
        v = validation.iloc[i]
        scores_row = marker_matrix.iloc[i].abs()
        best = int(scores_row.values.argmax())
        alternative = "–" if best == order[i] else f"varimax {best + 1}: {scores_row.iloc[best]:.2f}"
        rows.append([
            f"{c.label_v3_z} {c.name}", order[i] + 1, f"{abs(marker_matrix.iloc[i, order[i]]):.2f}", alternative,
            f"{variance[i]:.1%}", f"{float(c.published_variance):.1%}",
            number(v.spearman_lexile), number(v.published_r_drp), number(v.spearman_bt_easiness),
            number(v.mean_literary), number(v.mean_informational),
        ])
    r.append(table(["Component", "Varimax component", "Marker score", "Higher marker score elsewhere",
                    "Variance", "Published variance", "r Lexile", "Published r DRP", "r BT", "Literary",
                    "Informational"], rows) + "\n")

    r.append("## Strongest loadings per component\n")
    for i, c in enumerate(components.itertuples()):
        loadings = pd.Series(rotated[:, i], index=measure_columns)
        loadings = loadings[loadings.abs() >= 0.30].sort_values(key=abs, ascending=False)
        r.append(f"**{c.label_v3_z} {c.name}** (Varimax component {order[i] + 1}; markers: {c.marker_measures})\n")
        r.append(table(["Measure", "Loading"], [[m, number(l)] for m, l in loadings.items()]) + "\n")

    r.append("## Marker scores: every name on every Varimax component\n")
    r.append("Sign-adjusted mean marker loading before orienting the components; the assignment uses the absolute "
             "values. Bold: assigned component.\n")
    rows = []
    for i, (name, row) in enumerate(marker_matrix.iterrows()):
        rows.append([name] + [f"**{value:+.2f}**" if j == order[i] else f"{value:+.2f}"
                              for j, value in enumerate(row)])
    r.append(table(["Name"] + [f"Varimax {j}" for j in range(1, n_components + 1)], rows) + "\n")

    r.append("## Eigenvalues (unrotated PCA)\n")
    cumulative = np.cumsum(full.explained_variance_ratio_)
    r.append(f"Components with eigenvalue > 1 (Kaiser criterion): {int((full.explained_variance_ > 1).sum())}; "
             f"{n_components} are kept as in the original.\n")
    r.append(table(["Component", "Eigenvalue", "Explained variance", "Cumulative"],
                   [[i + 1, f"{full.explained_variance_[i]:.2f}", f"{full.explained_variance_ratio_[i]:.1%}",
                     f"{cumulative[i]:.1%}"] for i in range(15)]) + "\n")

    r.append(f"## Input measure pairs with |r| > {HIGH_CORRELATION}\n")
    r.append("Removed by Graesser et al. (2011) before their PCA; kept here (see OPEN_QUESTIONS.md).\n")
    r.append(table(["Measure", "Measure", "r"], [[a, b, number(c)] for a, b, c in pairs]) + "\n")
    return "\n".join(r)


if __name__ == "__main__":
    main()
