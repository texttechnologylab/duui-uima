"""Download the resources that Coh-Metrix needs but that may not be redistributed in Git or in the Docker image.

The files are written to one directory with the names the image expects (Dockerfile, DUUI_COH_METRIX_*_PATH); mount it at
/usr/src/app/src/main/resources/external (README, "Resources supplied by the user"). Each source is pinned (commit or file)
and checked against the SHA-256 it had when this script was written; a different file is an error. Converted files (xlsx to
csv) are written next to it. sources.json in the directory records source, licence and checksums of every file.

Not downloaded:
- GermaNet: academic licence, each user needs their own; mount it separately (README, "GermaNet").
- SUBTLEX-DE: its download page (crr.ugent.be/archives/534) is offline (2026-11-04); place the file yourself (format below).

With --clear the CLEAR corpus (development only: the reference corpus of the CLEAR Text Easability models) is downloaded too,
to --clear-dir (default: where the project keeps it). It is never part of the image.

The resources come with their own licences, mostly non-commercial; using them is your responsibility (src/main/resources/
LICENSES.md, and the sources below). Dependencies: Python 3.10+, openpyxl (xlsx files).

Usage: python download_resources.py <external_dir> [--clear [--clear-dir DIR]] [--force]
"""

import argparse
import csv
import datetime
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "duui-coh-metrix download_resources.py (https://github.com/texttechnologylab/duui-uima)"
CLEAR_DIR = Path("/storage/projects/CORE/projects2/coh_metrix/clear_corpus")

# name: file name in the external directory; url and sha256: the source as downloaded; convert: how the file is made from
# it ("copy", "zip:<member>", "xlsx:<sheet or index>:<delimiter>"); reference: the publication the file belongs to, and
# file: which file of it, so that it can be found again if the URL changes; licence: as stated by the source
RESOURCES = [
    {
        "name": "SUBTLEXus74286wordstextversion.txt",
        "url": "https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus2.zip",
        "sha256": "67e595da1b399d2a21e25a1466a8d9f242f21a219c6bbbd3e089c4779ad83856",
        "convert": "zip:SUBTLEXus74286wordstextversion.txt",
        "reference": "Marc Brysbaert and Boris New, \"Moving beyond Kučera and Francis: A critical evaluation of current word "
                     "frequency norms and the introduction of a new and improved word frequency measure for American "
                     "English\", Behavior Research Methods 41(4), 977-990, 2009, https://doi.org/10.3758/BRM.41.4.977",
        "file": "SUBTLEX-US, \"Zipped Text version with all 74,286 words in the corpus\" (subtlexus2.zip), Ghent University",
        "licence": "none stated on the download page",
        "used_for": "English WRDFRQc, WRDFRQa, WRDFRQmc, RDL2",
    },
    {
        "name": "DimLex.xml",
        "url": "https://raw.githubusercontent.com/discourse-lab/dimlex/8dbf87e9d1c8dc5664e838a2904d0a998200a83f/DimLex.xml",
        "sha256": "f04c6bc3b60c5b28567a6d8b4499dcf5a59a1444c4274eff9b151347547ebfdb",
        "convert": "copy",
        "reference": "Manfred Stede and Carla Umbach, \"DiMLex: A Lexicon of Discourse Markers for Text Generation and "
                     "Understanding\", COLING-ACL 1998, https://aclanthology.org/P98-2202/; Tatjana Scheffler and Manfred "
                     "Stede, \"Adding Semantic Relations to a Large-Coverage Connective Lexicon of German\", LREC 2016, "
                     "https://aclanthology.org/L16-1160/",
        "file": "DimLex.xml of github.com/discourse-lab/dimlex, commit 8dbf87e9",
        "licence": "CC BY-NC-SA 4.0 (LICENSE of github.com/discourse-lab/dimlex)",
        "used_for": "German CNC*, SMCAUSvp, SMCAUSr",
    },
    {
        "name": "en_dimlex.xml",
        "url": "https://raw.githubusercontent.com/discourse-lab/en_dimlex/2ecbab84f5e971d28abecc98780e7e39fb8bc931/"
               "en_dimlex.xml",
        "sha256": "5a6b8aa5823e696d980e04d993bc41dc581f8ab2a81ef4759664590b1b923d9f",
        "convert": "copy",
        "reference": "Debopam Das, Tatjana Scheffler, Peter Bourgonje and Manfred Stede, \"Constructing a Lexicon of English "
                     "Discourse Connectives\", SIGDIAL 2018, https://aclanthology.org/W18-5042/",
        "file": "en_dimlex.xml of github.com/discourse-lab/en_dimlex, commit 2ecbab84",
        "licence": "CC BY-NC-SA 4.0 (README of github.com/discourse-lab/en_dimlex; no LICENSE file)",
        "used_for": "English CNC*, SMCAUSvp, SMCAUSr",
    },
    {
        "name": "kuperman_aoa.csv",
        "url": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0348-8/MediaObjects/"
               "13428_2013_348_MOESM1_ESM.xlsx",
        "sha256": "3f69a1332359de1cd4a7ccd3c4c3c2e39b388eeb171d6e90544709c3dc1a8a6e",
        "convert": "xlsx:0:,",
        "reference": "Victor Kuperman, Hans Stadthagen-Gonzalez and Marc Brysbaert, \"Age-of-acquisition ratings for 30,000 "
                     "English words\", Behavior Research Methods 44(4), 978-990, 2012, "
                     "https://doi.org/10.3758/s13428-012-0210-4",
        "file": "the ratings (columns Word, ..., Rating.Mean; 30,121 words), Electronic Supplementary Material 1 (xlsx) of the "
                "erratum to the article, https://doi.org/10.3758/s13428-013-0348-8",
        "licence": "not stated",
        "used_for": "English WRDAOAc",
    },
    {
        "name": "brysbaert_concreteness.csv",
        "url": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5/MediaObjects/"
               "13428_2013_403_MOESM1_ESM.xlsx",
        "sha256": "1673ead761e28833a40e82c0d20f10782955ced9366d600eafeefee0f2254545",
        "convert": "xlsx:0:,",
        "reference": "Marc Brysbaert, Amy Beth Warriner and Victor Kuperman, \"Concreteness ratings for 40 thousand generally "
                     "known English word lemmas\", Behavior Research Methods 46(3), 904-911, 2014, "
                     "https://doi.org/10.3758/s13428-013-0403-5",
        "file": "the ratings (columns Word, Bigram, Conc.M, ...), Electronic Supplementary Material 1 (xlsx) of the article",
        "licence": "not stated",
        "used_for": "English WRDCNCc",
    },
    {
        "name": "13428_2016_718_MOESM1_ESM.csv",
        "url": "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-016-0718-0/MediaObjects/"
               "13428_2016_718_MOESM1_ESM.csv",
        "sha256": "604cd68e95272514324a5831269f06d014887fbad1e7ad763272a08a6a2f8eab",
        "convert": "copy",
        "reference": "Julia M. H. Birchenough, Robert Davies and Vincent Connelly, \"Rated age-of-acquisition norms for over "
                     "3,200 German words\", Behavior Research Methods 49(2), 484-501, 2017, "
                     "https://doi.org/10.3758/s13428-016-0718-0",
        "file": "the norms (columns Word, ..., AoAestimate), Electronic Supplementary Material 1 (csv) of the article",
        "licence": "not stated",
        "used_for": "German WRDAOAc",
    },
    {
        "name": "affective_norms.txt.gz",
        "url": "https://www.ims.uni-stuttgart.de/documents/ressourcen/experiment-daten/affective_norms.txt.gz",
        "sha256": "6696b4f2f4b8eaf1f6c9330caf8d7ad9a8f3a4bc75a2360c071bee5aba42c4b9",
        "convert": "copy",
        "reference": "Maximilian Köper and Sabine Schulte im Walde, \"Automatically Generated Affective Norms of Abstractness, "
                     "Arousal, Imageability and Valence for 350 000 German Lemmas\", LREC 2016, "
                     "https://aclanthology.org/L16-1413/",
        "file": "affective_norms.txt.gz (lemma, abstractness/concreteness, arousal, imageability, valence), IMS Stuttgart",
        "licence": "CC BY-NC-SA 4.0 (https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/affective-norms/)",
        "used_for": "German WRDCNCc, WRDIMGc",
    },
]

MANUAL = [
    {
        "name": "SUBTLEXde_SUBTLWF.csv",
        "reference": "Marc Brysbaert, Matthias Buchmeier, Markus Conrad, Arthur M. Jacobs, Jens Bölte and Andrea Böhl, \"The "
                     "Word Frequency Effect: A Review of Recent Developments and Implications for the Choice of Frequency "
                     "Estimates in German\", Experimental Psychology 58(5), 412-424, 2011, "
                     "https://doi.org/10.1027/1618-3169/a000123",
        "file": "SUBTLEX-DE, the word frequencies per million (SUBTLWF) of the article, formerly at crr.ugent.be/archives/534",
        "licence": "\"free for educational purposes\" (the article); download page offline (crr.ugent.be/archives/534)",
        "used_for": "German WRDFRQc, WRDFRQa, WRDFRQmc, RDL2",
        "format": "word TAB frequency per million, one word per line, no header (SUBTLEX-DE's SUBTLWF column)",
    },
]

CLEAR = {
    "name": "CLEAR_corpus_final.csv",
    "url": "https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/dbf1cf0df30bf6be8b0691d2c078b75043ca6199/"
           "CLEAR_corpus_final.xlsx",
    "sha256": "e87a19a29c12a441e0c4f11a4e49897ff24ac7dff240ee7397283a050a902cc7",
    "convert": "xlsx:Data:\t",
    "reference": "Scott Crossley, Aron Heintz, Joon Suh Choi, Jordan Batchelor, Mehrnoush Karimi and Agnes Malatinszky, \"A "
                 "large-scaled corpus for assessing text readability\", Behavior Research Methods 55(2), 491-507, 2023 (online "
                 "2022), https://doi.org/10.3758/s13428-022-01802-x",
    "file": "CLEAR_corpus_final.xlsx (sheet Data) of github.com/scrosseye/CLEAR-Corpus, commit dbf1cf0d; written tab-separated",
    "licence": "CC BY-NC-SA 4.0 (README of github.com/scrosseye/CLEAR-Corpus)",
    "used_for": "development only: reference corpus of the CLEAR Text Easability models (DUUIClearCorpusReader)",
}


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cell(value) -> str:
    """a spreadsheet value as text: empty for no value, floats with 15 significant digits (as spreadsheets export them)"""
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def xlsx_to_csv(data: bytes, sheet: str, delimiter: str) -> bytes:
    """one sheet (name, or index) as delimited text, UTF-8, LF; the stored values of formula cells"""
    import openpyxl
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    worksheet = workbook.worksheets[int(sheet)] if sheet.isdigit() else workbook[sheet]
    out = io.StringIO()
    writer = csv.writer(out, delimiter=delimiter, lineterminator="\n")
    for row in worksheet.iter_rows(values_only=True):
        writer.writerow(cell(v) for v in row)
    return out.getvalue().encode("utf-8")


def convert(data: bytes, how: str) -> bytes:
    if how == "copy":
        return data
    kind, _, rest = how.partition(":")
    if kind == "zip":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return archive.read(rest)
    if kind == "xlsx":
        sheet, _, delimiter = rest.partition(":")
        return xlsx_to_csv(data, sheet, delimiter)
    raise ValueError(how)


def fetch(resource: dict, directory: Path, force: bool) -> dict:
    target = directory / resource["name"]
    if target.exists() and not force:
        print(f"exists, skipped: {target}")
        return {**resource, "file_sha256": sha256(target.read_bytes()), "downloaded": None}
    print(f"downloading {resource['name']} from {resource['url']}")
    try:
        data = download(resource["url"])
    except OSError as e:
        raise SystemExit(f"{resource['url']}: {e}. Needed: {resource['file']}; {resource['reference']}")
    if sha256(data) != resource["sha256"]:
        raise SystemExit(f"{resource['url']}: SHA-256 {sha256(data)}, expected {resource['sha256']}; the source has changed, "
                         f"check it before using it (nothing written). Needed: {resource['file']}; {resource['reference']}")
    content = convert(data, resource["convert"])
    target.write_bytes(content)
    return {**resource, "file_sha256": sha256(content), "downloaded": datetime.date.today().isoformat()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("external_dir", type=Path, help="directory to mount as .../resources/external")
    parser.add_argument("--clear", action="store_true", help="also download the CLEAR corpus (development only)")
    parser.add_argument("--clear-dir", type=Path, default=CLEAR_DIR, help=f"default: {CLEAR_DIR}")
    parser.add_argument("--force", action="store_true", help="download again even if a file exists")
    args = parser.parse_args()

    args.external_dir.mkdir(parents=True, exist_ok=True)
    records = [fetch(r, args.external_dir, args.force) for r in RESOURCES]
    for manual in MANUAL:
        target = args.external_dir / manual["name"]
        records.append({**manual, "file_sha256": sha256(target.read_bytes()) if target.exists() else None})
        if not target.exists():
            print(f"\nNOT DOWNLOADED: {manual['name']}: {manual['file']}; {manual['reference']}. {manual['licence']}.\n"
                  f"  Place it yourself in {args.external_dir}; format: {manual['format']}. "
                  f"Without it: {manual['used_for']} are None.", file=sys.stderr)
    (args.external_dir / "sources.json").write_text(json.dumps(records, indent=1, ensure_ascii=False) + "\n")

    if args.clear:
        args.clear_dir.mkdir(parents=True, exist_ok=True)
        clear = fetch(CLEAR, args.clear_dir, args.force)
        (args.clear_dir / "CLEAR_corpus_final.sources.json").write_text(json.dumps(clear, indent=1, ensure_ascii=False) + "\n")

    print(f"\nDone: {args.external_dir} (sources.json lists sources, licences and checksums). GermaNet is not downloaded "
          f"(own licence; README, \"GermaNet\"). Mount the directory at /usr/src/app/src/main/resources/external.")


if __name__ == "__main__":
    main()
