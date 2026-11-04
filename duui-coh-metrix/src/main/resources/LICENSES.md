# Licences of the resources of duui-coh-metrix

The code is under the AGPL-3.0 (repository `LICENSE`). The files below are separate works that keep their own licences; the
AGPL does not extend to them.

## In the image

| file | source | licence |
|---|---|---|
| `mrc_psycholinguistic_database.csv` (downloaded at build) | MRC Psycholinguistic Database (Coltheart 1981; Wilson 1988), Oxford Text Archive item 1054 | CC BY-NC-SA 3.0: non-commercial use only, attribution, same licence |
| `mrc_psycholinguistic_database_de.csv` | German translation of the MRC database (Texttechnology Lab), an adaptation of the above | CC BY-NC-SA 3.0, as its source |
| `nltk_data/` (downloaded at build) | Princeton WordNet 3.0; Open Multilingual Wordnet 1.4; CMU Pronouncing Dictionary | WordNet 3.0 licence; the licences listed in the OMW archive; BSD-style (Carnegie Mellon University) |
| `word_frequencies_{de,en}_*wiki-20220301-sample10000.csv` | word counts from Wikipedia samples | derived from Wikipedia, CC BY-SA |
| `gum_weights_en.csv` | how often 22 ambiguous English adverbs are connectives, counted on the GUM corpus annotations (Zeldes 2017) | GUM annotations CC BY 4.0; cite GUM |
| `easability/text_easability_*_ref*.json` | Text Easability models fitted on the reference corpora of coh-metrix_resurrection | as this repository |
| `depedit/stan2uni.ini` | DepEdit's Stanford Dependencies -> UD rules, unchanged: https://github.com/amir-zeldes/depedit, `examples/stan2uni.ini` (last changed in commit 96ad44c, 2017-12-19; SHA-256 `30d5da07…3208e14`, fetched 2026-11-03); Peng & Zeldes, "All Roads Lead to UD: Converting Stanford and Penn Parses to English Universal Dependencies with Multilayer Annotations", LAW-MWE-CxG 2018, https://aclanthology.org/W18-4918/ | Apache-2.0 (repository licence, copy in `depedit/LICENSE-depedit.txt`) |
| `depedit/clearnlp_additions.ini` | this component (ClearNLP labels that Stanford Dependencies does not have) | AGPL-3.0 |
| `TypeSystem.xml` | this component | AGPL-3.0 |

## Supplied by the user (README, "Resources supplied by the user")

Not in the image: GermaNet (academic licence, no redistribution), DiMLex and DiMLex-Eng (CC BY-NC-SA 4.0), the Köper & Schulte
im Walde affective norms (CC BY-NC-SA 4.0), and the lists without a licence for redistribution: SUBTLEX-DE, SUBTLEX-US, the
Birchenough et al. (2016) and Kuperman et al. (2012) age-of-acquisition norms, the Brysbaert et al. (2014) concreteness norms.
Licence check of 2026-11-02 (not legal advice).
