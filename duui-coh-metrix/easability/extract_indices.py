"""Collect the Coh-Metrix indices of a directory of DUUI output XMIs into one CSV.

One row per document: the document id, all MetaDataStringFields (for CLEAR the corpus columns, see
DUUIClearCorpusReader) and one column per index, keyed by labelTTLab (NaN stays NaN).

Usage: python extract_indices.py <xmi_dir> <output.csv>
"""

import argparse
import gzip
from pathlib import Path

import cassis
import pandas as pd

INDEX_TYPE = "org.texttechnologylab.uima.type.cohmetrix.Index"
METADATA_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData"
METADATA_FIELD_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.metadata.type.MetaDataStringField"


def read_document(path: Path, typesystem: cassis.TypeSystem) -> dict:
    with gzip.open(path) as f:
        cas = cassis.load_cas_from_xmi(f, typesystem=typesystem)
    row = {"document_id": cas.select(METADATA_TYPE)[0].get("documentId")}
    for field in cas.select(METADATA_FIELD_TYPE):
        row["meta_" + field.get("key")] = field.get("value")
    for index in cas.select(INDEX_TYPE):
        label = index.get("labelTTLab")
        if label in row:
            raise ValueError(f"{path}: duplicate index label {label}")
        row[label] = index.get("value")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("xmi_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with gzip.open(args.xmi_dir / "TypeSystem.xml.gz") as f:
        typesystem = cassis.load_typesystem(f)

    paths = sorted(args.xmi_dir.glob("*.xmi.gz"))
    rows = []
    for i, path in enumerate(paths, 1):
        rows.append(read_document(path, typesystem))
        if i % 500 == 0:
            print(f"{i}/{len(paths)}")

    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} documents, {table.shape[1]} columns -> {args.output}")


if __name__ == "__main__":
    main()
