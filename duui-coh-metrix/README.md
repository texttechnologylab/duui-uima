# Coh-Metrix Text Cohesion Analysis [WIP]

DUUI implementation of Coh-Metrix 3.0 for English and German. The component
computes 117 indices (the 106 of Coh-Metrix 3.0 and 11 project-only readability
formulas) over pre-annotated UIMA documents; with the variants of other methods
(the textstat library, the CLEAR Text Easability models), a German document gets
up to 145 records, an English one up to 138. The indices include
descriptive, referential-cohesion, LSA, lexical-diversity, connective,
situation-model, syntactic-complexity, word-information, and readability
measures. Every index carries its Coh-Metrix 3.0 label and an internal label
that names the resources used where they differ from or replace those of
Coh-Metrix 3.0 (see [Index labels](#index-labels)).

## Prerequisites and preprocessing

The annotator requires a **pre-annotated UIMA CAS** (document language `de` or
`en`) with:

- `de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph`;
- sentences: `org.texttechnologylab.uima.type.stanza.StanzaSentence` if
  present, else every `Sentence`;
- tokens, exactly one layer, in this order of preference:
  `org.texttechnologylab.uima.type.stanza.StanzaToken`, `SpacyToken`, plain
  `Token`. Each with lemma, POS (`posValue` STTS / Penn Treebank,
  `coarseValue` UPOS), morphological features (including `Number[psor]`,
  `VerbForm`, `Mood`) and dependencies in Universal Dependencies v2. Input with
  spaCy's own labels is mapped (see [Best effort without UD](#best-effort-without-ud)).
  Noun chunks are built from the UD tree;
- token vectors as `WordEmbedding` (see [Vectors](#vectors)), for the `LSA*`
  indices and `SMCAUSlsa`. If no usable vector or comparison can be
  constructed, the affected result is `None`.

The default pipeline is the DUUI form of the reference preprocessing of
Coh-Metrix 0.4.0-py (its defaults per language):

1. paragraphs: `duui-chonky` 1.1.0 with `keep_line_breaks=filter` (German), or
   `duui-wtpsplit` 1.1.0 with `segment=paragraphs` (English);
2. sentences: `duui-wtpsplit` within each paragraph,
   `split_on_input_newlines=true`;
3. tokens: `duui-somajo` 1.0.0 within each sentence (`de_CMC` / `en_PTB`);
4. `duui-stanza` 1.0.0 (`de|en/default_accurate`): StanzaToken, StanzaSentence,
   POS, lemma, morphology, UD dependencies; German contractions split into two
   words ("zum" → "zu" + "dem");
5. `duui-spacy` 0.9.0 with `model_variant=large`, `use_existing_tokens=true`,
   `use_existing_sentences=true`,
   `token_type=org.texttechnologylab.uima.type.stanza.StanzaToken`,
   `write_types=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token`: a
   `SpacyToken` and a `WordEmbedding` (`SPACYlg`) per Stanza word, and nothing
   else of spaCy's (no second POS, lemma, morphology, parse or entities; about a
   third fewer annotations; its noun chunks are written anyway and not read);
6. `duui-remove-annotations` 1.0.0 with
   `types=de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token`,
   `exact=true` (the plain layers of steps 2 and 3);
7. Coh-Metrix.

`CohMetrixUdPipelineTest` runs this pipeline. On 18 texts (10 German, 8
English), the pipeline and the reference preprocessing gave the same
paragraphs, sentences, words, annotations and vectors, and all 2,870 index
values were identical (2026-11-04).

A spaCy-only pipeline (e.g. `duui-somajo` → `duui-spacy` → remover →
Coh-Metrix) still runs, as best effort (see below).

## Best effort without UD

The syntactic rules read Universal Dependencies version 2 (e.g. from
`duui-stanza`). Input with spaCy's labels is mapped to UD in the component, as
far as that is possible without a UD parser; the input in the CAS is not
changed. The scheme is detected from the labels (`ud_mapping.py`):

| Input | Mapping | Group |
|---|---|---|
| UD v2 | none | none |
| German TIGER (spaCy `de_core_news_*`) | renaming table by label and part of speech; the TIGER tree is kept (the finite auxiliary, the preposition and the copula stay heads) | `TIGERmap` |
| English ClearNLP (spaCy `en_core_web_*`) | [DepEdit](https://github.com/amir-zeldes/depedit) with its Stanford Dependencies to UD rules (`stan2uni.ini`; Siyao Peng and Amir Zeldes, ["All Roads Lead to UD: Converting Stanford and Penn Parses to English Universal Dependencies with Multilayer Annotations"](https://aclanthology.org/W18-4918/), LAW-MWE-CxG 2018), after our rules for the ClearNLP labels Stanford Dependencies does not have (`clearnlp_additions.ini`: the copula as head of `attr` / `acomp`, `relcl`, `dative`, `agent`, `oprd`); relabels and restructures the tree | `CLEARNLPmap` |
| UD v1, other schemes | refused (HTTP 422) | |

The records that read the dependency relations get the group after the public
label, e.g. `SYNNP_TIGERmap`, `SMCAUSr_CLEARNLPmap_WN_DIMLEXeng_GUM`: the
connectives (`CNC*` with DiMLex), `DRAP`, `DRGERUND`, `DRINF`, `DRNEG`, `DRNP`,
`DRPP`, `DRPVAL`, `DRVP`, `SMCAUSv`, `SMCAUSvp`, `SMCAUSr`, `SMINTEp`,
`SMINTEr`, `SMTEMP`, `SYNLE`, `SYNNP`, `SYNSTRUTa`, `SYNSTRUTt` and `RDL2`
(`DEPENDENCY_KEYS` in `duui.py`). The Text Easability components (`PC*`) are
`None` with an error: their models are fitted on UD input.

**Known limits.** English (restructured) comes close to UD input; e.g. SYNNP
0.61/0.94/0.72 on three short test texts with spaCy against 0.62/0.86/0.62
with Stanza. German keeps the TIGER tree, so the indices that need UD structure
are clearly off: SYNNP 0.39/0.75/0.56 against 0.90/1.00/0.67 with Stanza (the
preposition stays the head of its noun), DRPVAL is always 0 (the auxiliary
stays the head, so there is no `aux:pass`), RDL2 differs too. The segmentation
also differed in these comparisons (spaCy's own sentences against regular
expressions for Stanza), so the numbers are no clean measurement of the mapping
alone. Use the default pipeline (Stanza) for German.

## Resources

The resources come in four kinds, depending on whether they may be
redistributed (`src/main/resources/LICENSES.md`):

| Kind | Resources | In Git | In the image | What you do |
|---|---|---|---|---|
| Our own, or redistributable | Wikipedia frequency samples, German MRC translation, GUM weights, Text Easability models, DepEdit rules | yes | yes | nothing |
| Redistributable, downloaded when the image is built | English MRC database, NLTK data (WordNet, Open Multilingual Wordnet, CMU Pronouncing Dictionary) | no | yes | nothing |
| Not redistributable, but downloadable | SUBTLEX-US, SUBTLEX-DE, DiMLex, DiMLex-Eng, Kuperman, Brysbaert, Birchenough, Köper & Schulte im Walde | no, never | no, never | download with `download_resources.py`, mount the directory |
| Licensed to you only | GermaNet | no, never | no, never | obtain it yourself, mount it |

Two mounts (see [Start the Docker container](#start-the-docker-container)):

| Mount point in the container | Content | Environment variables |
|---|---|---|
| `/usr/src/app/src/main/resources/external` | the files of [Resources supplied by the user](#resources-supplied-by-the-user), flat, with the names given there | `DUUI_COH_METRIX_<RESOURCE>_PATH`, one per file |
| `/usr/src/app/src/main/resources/germanet` | the GermaNet XML directory (e.g. `GN_V180_XML` of GermaNet 18.0) | `DUUI_COH_METRIX_GERMANET_PATH` |

Without a mount the component runs. A record with a **fallback** then uses it
instead of the missing resource, under its own internal label (label rule 7):
the connectives and causal particles use LLM-generated lists (`CNCAll_LLMlist`
instead of `CNCAll_DIMLEX`), the frequencies the Wikipedia samples
(`WRDFRQc_WIKI` instead of `WRDFRQc_SUBTLEX`), the norms the MRC database
(`WRDAOAc_MRC` instead of `WRDAOAc_KUPERMAN`), and the German causal and
intentional verbs seed lists (see [GermaNet](#germanet)). A fallback is computed
only then, unless all records are requested (`all_records=true`, see
[Selecting indices](#selecting-indices)). The other records that need a missing
resource are `None` with the error `resource not available: ...`.

### In the image

| Resource | Used for | Source and licence (`src/main/resources/LICENSES.md`) |
|---|---|---|
| MRC Psycholinguistic Database (`mrc_psycholinguistic_database.csv`, downloaded at build time) | English WRDFAMc, WRDIMGc, WRDMEAc; fallback `_MRC` of WRDAOAc, WRDCNCc | see below |
| MRC translated into German (`mrc_psycholinguistic_database_de.csv`) | German WRDFAMc, WRDMEAc; fallback `_MRCtr` of WRDAOAc, WRDCNCc, WRDIMGc | see below |
| Wikipedia frequency samples (`word_frequencies_{de,en}_*wiki-20220301-sample10000.csv`) | fallback `_WIKI` of WRDFRQ*, RDL2 (without SUBTLEX) | see "Word frequencies" |
| GUM weights (`gum_weights_en.csv`) | English connectives (`DIMLEXeng_GUM`) | counted on the GUM annotations: Amir Zeldes, ["The GUM corpus: creating multilayer resources in the classroom"](https://doi.org/10.1007/s10579-016-9343-x), Language Resources and Evaluation 51(3), 581–612, 2017; annotations CC BY 4.0 (GUM's `LICENSE.md`); the table holds counts, no text |
| Text Easability models (`easability/*.json`) | PC* | see "Text Easability models" |
| Princeton WordNet 3.0, Open Multilingual Wordnet 1.4, CMU Pronouncing Dictionary (NLTK data, downloaded at build time) | English WRDPOLc, WRDHYP*, SMCAUSv, SMINTEp, SMCAUSwn, SMCAUSvp, SMCAUSr; English syllables | NLTK data |
| DepEdit rules (`depedit/`) | best effort, English | see "Best effort without UD" |

### Resources supplied by the user

These files are not in the image (non-commercial licences, or no licence for
redistribution found) and must never be added to Git or the image.
`download_resources.py` downloads them into one directory, with the names the
image expects, checks each file against the SHA-256 it had when the script was
written, and writes `sources.json` (source, publication, licence, checksums):

```
pip install openpyxl
python download_resources.py /path/to/external
```

Use a directory outside the repository and mount it at
`/usr/src/app/src/main/resources/external`; each file
has its own environment variable (Dockerfile) if another name or place is
needed. `--clear` also downloads the CLEAR corpus, only needed to refit the
CLEAR Text Easability models (development). Without a file, its records use
their fallback (see above) or are `None` with the error `resource not
available: ...`; the Text Easability components that need it as input are
`None` (the models take only inputs from the resources they were fitted on).

| File (environment variable `DUUI_COH_METRIX_..._PATH`) | Resource | Used for |
|---|---|---|
| `SUBTLEXde_SUBTLWF.csv` (`SUBTLEX_DE`) | SUBTLEX-DE word frequencies per million; tab-separated, word and frequency per line, no header. Marc Brysbaert, Matthias Buchmeier, Markus Conrad, Arthur M. Jacobs, Jens Bölte and Andrea Böhl, ["The Word Frequency Effect: A Review of Recent Developments and Implications for the Choice of Frequency Estimates in German"](https://doi.org/10.1027/1618-3169/a000123), Experimental Psychology 58(5), 412–424, 2011. The file is the authors' release `SUBTLEX-DE_txt_cleaned_with_Google00.zip` of crr.ugent.be/SUBTLEX-DE/ (offline; [Internet Archive snapshot](https://web.archive.org/web/2019/http://crr.ugent.be/subtlex-de/)): all 190,500 word forms, column `SUBTLEX`, converted to the format above. Not CTAP's table (`SUBTLEXde_SUBTLWF.csv` of [ctapweb/ctap-multilingual](https://github.com/ctapweb/ctap-multilingual)), a subset of 65,499 of these forms by an undocumented rule: with it, rare words are not found and drop out of the means, so the German frequency indices and RDL2 come out higher. Licence not stated on the download page; the article: "free for educational purposes" | German WRDFRQc, WRDFRQa, WRDFRQmc, RDL2 |
| `SUBTLEXus74286wordstextversion.txt` (`SUBTLEX_US`) | SUBTLEX-US, column `SUBTLWF` of the text version. Marc Brysbaert and Boris New, ["Moving beyond Kučera and Francis: A critical evaluation of current word frequency norms and the introduction of a new and improved word frequency measure for American English"](https://doi.org/10.3758/BRM.41.4.977), Behavior Research Methods 41(4), 977–990, 2009; [download](https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/overview.htm) ("Zipped Text version with all 74,286 words"; no licence stated) | English WRDFRQc, WRDFRQa, WRDFRQmc, RDL2 |
| `DimLex.xml` (`DIMLEX`) | DiMLex, lexicon of German discourse connectives. Manfred Stede and Carla Umbach, ["DiMLex: A Lexicon of Discourse Markers for Text Generation and Understanding"](https://aclanthology.org/P98-2202/), COLING-ACL 1998; Tatjana Scheffler and Manfred Stede, ["Adding Semantic Relations to a Large-Coverage Connective Lexicon of German"](https://aclanthology.org/L16-1160/), LREC 2016; [github.com/discourse-lab/dimlex](https://github.com/discourse-lab/dimlex), CC BY-NC-SA 4.0 | German CNC*, SMCAUSvp, SMCAUSr |
| `en_dimlex.xml` (`DIMLEX_EN`) | DiMLex-Eng. Debopam Das, Tatjana Scheffler, Peter Bourgonje and Manfred Stede, ["Constructing a Lexicon of English Discourse Connectives"](https://aclanthology.org/W18-5042/), SIGDIAL 2018; [github.com/discourse-lab/en_dimlex](https://github.com/discourse-lab/en_dimlex), CC BY-NC-SA 4.0 (README) | English CNC*, SMCAUSvp, SMCAUSr |
| `kuperman_aoa.csv` (`KUPERMAN`) | Age-of-acquisition ratings, columns `Word`, `Rating.Mean` (years). Victor Kuperman, Hans Stadthagen-Gonzalez and Marc Brysbaert, ["Age-of-acquisition ratings for 30,000 English words"](https://doi.org/10.3758/s13428-012-0210-4), Behavior Research Methods 44(4), 978–990, 2012 (download page and licence not verified, 2026-11-04) | English WRDAOAc |
| `brysbaert_concreteness.csv` (`BRYSBAERT`) | Concreteness ratings, columns `Word`, `Conc.M` (1–5; rows with `Bigram` other than 0 are skipped). Marc Brysbaert, Amy Beth Warriner and Victor Kuperman, ["Concreteness ratings for 40 thousand generally known English word lemmas"](https://doi.org/10.3758/s13428-013-0403-5), Behavior Research Methods 46(3), 904–911, 2014; the article's supplementary file 1 (Excel, saved as CSV); licence not verified | English WRDCNCc |
| `13428_2016_718_MOESM1_ESM.csv` (`BIRCHENOUGH`) | Rated age-of-acquisition norms, columns `Word`, `AoAestimate` (years; Latin-1). Julia M. H. Birchenough, Robert Davies and Vincent Connelly, ["Rated age-of-acquisition norms for over 3,200 German words"](https://doi.org/10.3758/s13428-016-0718-0), Behavior Research Methods 49(2), 484–501, 2017; the article's supplementary file 1; licence not verified | German WRDAOAc |
| `affective_norms.txt.gz` (`KOEPER`) | Automatically extended norms, columns abstractness/concreteness and imageability (0–10). Maximilian Köper and Sabine Schulte im Walde, ["Automatically Generated Affective Norms of Abstractness, Arousal, Imageability and Valence for 350 000 German Lemmas"](https://aclanthology.org/L16-1413/), LREC 2016; [download](https://www.ims.uni-stuttgart.de/en/research/resources/experiment-data/affective-norms/), CC BY-NC-SA 4.0 | German WRDCNCc, WRDIMGc |

### GermaNet

German lexical-semantic indices use GermaNet through `germanetpy==0.2.5`.
GermaNet is not distributed with the image or the repository because it is a
separately licensed resource: each user needs their own licence from the
University of Tübingen
([GermaNet](https://uni-tuebingen.de/en/142806)). Mount its XML directory
(e.g. `GN_V180_XML`) at `/usr/src/app/src/main/resources/germanet`, or set
`DUUI_COH_METRIX_GERMANET_PATH` to where it is mounted.

Without GermaNet:

- German causal and intentional verbs come from hand-written seed lists (11
  causal, 12 intentional verbs), named in the label: `SMCAUSv_SEEDlist`
  instead of `SMCAUSv_GN`.
- GermaNet-dependent polysemy, hypernymy, and semantic verb-overlap results are
  `None`.
- German `SMTEMP` is computed from the UD annotations (tense and mood features,
  auxiliaries and participles). GermaNet is only needed for `am`/`beim` progressive constructions whose noun
  is not tagged as a verb, to decide whether the noun is a nominalized verb.
  Without GermaNet, the sentence pairs that contain such a sentence are
  excluded, the label drops `GN` (`SMTEMP_LLMlist`), and the index is `None`
  only if no valid pair remains.

English lexical-semantic indices use WordNet instead and do not require the
GermaNet mount.

### MRC psycholinguistic ratings

The English resource
`mrc_psycholinguistic_database.csv` is downloaded during the Docker build from
the public
[MRC Psycholinguistic Database dataset](https://huggingface.co/datasets/StephanAkkerman/MRC-psycholinguistic-database).
The implementation reads the ratings for age of acquisition, familiarity,
concreteness, imageability, and Colorado meaningfulness. The public source
contains approximately 150,000 lexical records; 9,335 entries have at least
one available target rating and therefore provide usable coverage for these
indices. Where the source has several records of a word, the first
non-missing rating is used.

`mrc_psycholinguistic_database_de.csv` is a project-specific German derivative.
The English headwords were translated with DeepL, normalized to German lemmas,
and duplicate translations were merged by averaging the available ratings.
The current resource contains
4,275 non-empty unique German entries. It is therefore an approximation and
not an independent German norming study. In both MRC files, a rating of `0` is
treated as the database's missing-value marker and is not included in averages.

Since 0.6.0 the norms are looked up by lemma (exact case first, then lower
case; German also with old spellings and "ss" for "ß"). Age of acquisition and
concreteness (both languages) and German imageability come from the native
norms above; MRC is their fallback (`_MRC` / `_MRCtr`), computed only if these
norms are not mounted. The norms are on
different scales (MRC 100–700, Kuperman and Birchenough years, Brysbaert 1–5,
Köper 0–10), so values of different resources are not comparable.

### Word frequencies

WRDFRQc, WRDFRQa and WRDFRQmc are the mean (WRDFRQmc: the mean per sentence of
the minimum) of the **base-10 logarithm** of the frequency per million. Main
record: SUBTLEX (SUBTLEX-US, SUBTLEX-DE, see above). Words not in the list are
left out rather than counted with frequency 0.

The fallback `_WIKI`, computed only if SUBTLEX is not mounted, uses the files
`word_frequencies_en_enwiki-20220301-sample10000.csv` and
`word_frequencies_de_dewiki-20220301-sample10000.csv`: project resources
derived from 10,000-article samples of the English and German Wikipedia dumps
dated 2022-03-01, normalized to frequencies per million, with the same
logarithm.

Both replace the non-public CELEX data of the original Coh-Metrix, so the
absolute values are not comparable with CELEX-based Coh-Metrix output. The
logarithm deviates from the book, which describes WRDFRQc as the raw mean (see
[Index definitions and deviations](#index-definitions-and-deviations)).

### Syllable counts

- German: the vowel nuclei of the spelling, with ä, ö, ü and y as vowels,
  diphthongs and double vowels as one nucleus, hyphenated words counted by
  their parts, and short words with several capitals read letter by letter
  ("USA" 3, "AfD" 3). This rule comes from Coh-Metrix 0.4.0-py (Maxim Konca)
  and replaced Pyphen, which never splits off a single vowel ("Abend" 1).
  Against syllable nuclei counted from the German Wiktionary's IPA it is exact
  for 98.1 % of the tokens of his reference texts (Pyphen 93.6 %); this is his
  measurement, not verified by us.
- English: the CMU Pronouncing Dictionary, else a spelling rule; a clitic
  ("n't", "'s") counts what it adds to its host.
- The second word of a split German contraction counts 0 syllables and 0
  letters ("im" = "in" + "dem" counts as "im").

### Readability formulas and textstat

The main records of the readability formulas (`RDFRE`, `RDFKGL` and the
project-only `RD*_ext`) are computed by this component from its own units: the
sentences of DESSC, the words of DESWC, the syllables of DESWLsy, the letters of
DESWLlt. So RDFRE and RDFKGL follow from DESSL and DESWLsy as Coh-Metrix 3.0
defines them. Constants, thresholds and easy-word lists are those of textstat
0.7.12; German RDFRE uses Amstad's constants (180, 1, 58.5). Fog, Dale-Chall and
Spache count "difficult words" with textstat's per-language easy-word list, so
their labels carry `TEXTSTATlist` (`RDFOG_ext_TEXTSTATlist`).

The variant `_TEXTSTAT` (e.g. `RDFRE_TEXTSTAT`) is computed by the textstat
library itself, the fork
[felixvoll/textstat](https://github.com/felixvoll/textstat/tree/cohmetrix-de),
tag `v0.7.12-post2` (apart from a German easy-word list and a UTF-8 fix for
loading it, identical to textstat 0.7.12). textstat gets the raw text and does
its own sentence and word splitting, so its counts can differ from the main
records:

| Index                           | Language-specific part in textstat                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------------------------------|
| RDFRE                           | German uses Amstad's constants (180, 1, 58.5) instead of Flesch's                                           |
| RDFOG_ext, RDDCRS_ext, RDSPACHE_ext | "Difficult words" from a per-language easy-word list; the German list (2,579 words, the fork's addition) is a translation and manual adaptation of the English Dale-Chall/Spache list |
| RDFKGL, RDSMOG_ext, RDLW_ext, RDWSTF1–4_ext | Only syllable counting: Pyphen for German, the CMU Pronouncing Dictionary with Pyphen as fallback for English |
| RDARI_ext, RDCLI_ext            | None (characters, words, sentences)                                                                         |

### Text Easability models

The Text Easability components (`PC*`, indices 12–27) are z scores and
percentiles of a principal component analysis. The original components
(Arthur C. Graesser, Danielle S. McNamara and Jonna M. Kulikowich,
["Coh-Metrix: Providing Multilevel Analyses of Text Characteristics"](https://doi.org/10.3102/0013189X11413260),
Educational Researcher 40(5), 223–234, 2011) come from the non-public TASA
corpus, so the models here are fitted again on other reference corpora:

| Records | Model | Reference corpus |
|---|---|---|
| `PC*_REFIT_REFa` (English), `PC*_REFIT_REFv1` (German), main records | `text_easability_en_refa-en-rr-gn-cr5.json`, `text_easability_de_refv1-ud-rr-cr5.json` (Maxim Konca, fitted on the values of this component, 2026-11-05) | 9 registers × 1,000 passages of 250–350 words each (news report, personal narrative, encyclopedia, informational description, research and expert, how-to, opinion/review/advice, persuasion/commercial, fiction). English: FineWeb sample-10BT, wiki40b, Wikinews, Intereconomics, fiction 1900–1918 from PG-19; German: FineWeb-2 and other web sources, fiction 1900–1930 from de-Corp. PCA on standardized inputs with Varimax rotation; the English components are named by their congruence with the German model |
| `PC*_REFIT_CLEAR` (English), `PC*_REFIT_CLEARtr` (German), variants | `text_easability_en_clear.json`, `text_easability_de_cleartr.json` | the [CommonLit Ease of Readability (CLEAR) corpus](https://github.com/scrosseye/CLEAR-Corpus) (Crossley et al., 2022), 4,724 excerpts; German: CLEAR translated into German (Kolibri-1 Q4_K_M, 4,669 excerpts without truncated translations and paragraph mismatches) |

**Both are computed since 0.7.0.** Both are fitted on the values of this
component (0.6.0; 0.7.0 computes the same index values): REFa/REFv1 as cr5
(the English cr5 refitted on reference corpus A parsed with the full ELECTRA
tokenizer, the German cr5 equal to cr4), CLEAR on 2026-11-04 (see below). In
0.6.0 they were `None` with an error: the main models of 0.5.0-py had other
SYNNP, LSAGN and German SMTEMP values as inputs (`MODEL_INPUTS_CHANGED` in
`duui.py`, empty since cr4), the CLEAR models those of 0.2.0. On the same CLEAR
texts the two model families agree for most components (z score correlation
0.80–0.99), except German Temporality and, in English, Verb cohesion and
Temporality: there the English cr5 names the components the other way round
(its "Semantic overlap", PCVERB, is the CLEAR model's PCTEMP; its "Third-person
narration", PCTEMP, is the CLEAR model's PCVERB with the opposite sign).

How the components are computed: the inputs are standardized with the reference
corpus' mean and SD, the z score is the weighted sum, the percentile the share
of reference texts with a lower z. An input that is also undefined in some
reference texts (e.g. causal cohesion without causal verbs) counts as the
reference mean; if any other input is undefined (empty document, single
sentence), the components are `None`. The models only take inputs computed with
the resources they were fitted on. The component descriptions use the names of
Coh-Metrix 3.0; where a refitted component carries other markers, the model's
own name is added, e.g. "Text Easability PC Verb cohesion (in this model:
Semantic overlap), z score" (provisional).

**Refitting the CLEAR variants.** The corpus is not part of this repository:
`python download_resources.py <external_dir> --clear` writes
`CLEAR_corpus_final.csv` to the corpus directory (`CLEAR_DIR` in
`ClearCorpusCohMetrixTest`, `/storage/projects/CORE/projects2/coh_metrix/clear_corpus`;
`$CLEAR` below). `ClearCorpusCohMetrixTest` runs the default pipeline on
CLEAR's own paragraphs (one per line) and writes one XMI per excerpt to
`$CLEAR/xmi_en_0.6.0` (German, from the translation:
`$CLEAR/xmi_de_kolibri-1-q4km_0.6.0`). It needs the services of the default
pipeline (system properties as in `CohMetrixUdPipelineTest`), Coh-Metrix 0.6.0
with GermaNet and the resources supplied by the user mounted: they are model
inputs. The PCA inputs are the main records of the indices
(`easability/pca_input_measures.csv`, internal labels per language in
`labels_en` / `labels_de`; written to the model as `labels_internal`). The PCA
scripts use the dependencies in `requirements-dev.txt` (Python 3.12):

```bash
CLEAR=/storage/projects/CORE/projects2/coh_metrix/clear_corpus
URLS="-Dwtpsplit.url=... -Dsomajo.url=... -Dstanza.url=... -Dspacy.url=... -Dremove.url=... -Dcohmetrix.url=..."
mvn test -Dtest='ClearCorpusCohMetrixTest#runPipeline' $URLS
python easability/extract_indices.py $CLEAR/xmi_en_0.6.0 $CLEAR/indices_en_0.6.0.csv
python easability/pca.py $CLEAR/indices_en_0.6.0.csv easability/results/clear_en_pca \
    --model src/main/resources/easability/text_easability_en_clear.json --language en \
    --corpus "CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, 4724 excerpts; duui-coh-metrix 0.6.0, default pipeline" \
    --label-suffix _REFIT_CLEAR
```

`pca.py` writes the model and, to `easability/results/clear_en_pca/`, a
readable summary (`report.md`: comparison with the published components,
loadings, marker scores, validation against the CLEAR difficulty ratings) and
the underlying tables, including the component scores of every CLEAR excerpt
(`scores.csv`, by excerpt ID). German: place the translation in `$CLEAR`
(default `translation_kolibri-1-q4km/`, other versions with
`-Dclear.translation=...`):

```bash
mvn test -Dtest='ClearCorpusCohMetrixTest#runPipelineGermanTranslation' $URLS
python easability/extract_indices.py $CLEAR/xmi_de_kolibri-1-q4km_0.6.0 \
    $CLEAR/indices_de_kolibri-1-q4km_0.6.0.csv
python easability/pca.py $CLEAR/indices_de_kolibri-1-q4km_0.6.0.csv \
    easability/results/clear_de_kolibri-1-q4km_pca \
    --model src/main/resources/easability/text_easability_de_cleartr.json --language de \
    --corpus "CLEAR corpus (Crossley et al. 2022), CLEAR_corpus_final.csv, translated into German with Kolibri-1 Q4_K_M (translation_kolibri-1-q4km), excerpts without truncated translations and paragraph mismatches; duui-coh-metrix 0.6.0, default pipeline" \
    --label-suffix _REFIT_CLEARtr
python easability/compare_models.py easability/results/clear_en_pca \
    easability/results/clear_de_kolibri-1-q4km_pca \
    easability/results/clear_en_vs_de_kolibri-1-q4km --name-a EN --name-b DE
```

Test runs: `-Dclear.limit=N` processes only the first N excerpts,
`-Dclear.target=DIR` writes to another directory. The models in the repository
and the results in `easability/results/` are those of the refit on 0.6.0
(2026-11-04): English 4,724 excerpts, 49 measures, 63.3% of the variance
explained by the 8 components; German 4,669 excerpts, 49 measures, 62.6%. In
both, Temporality does not fit its markers (marker score 0.09 English, 0.11
German; `report.md` in each results directory).
`ClearCorpusTextEasabilityTest` checks that the component's `_REFIT_CLEAR(tr)`
records reproduce the PCA scores (every 100th excerpt of the reference corpus,
against the running service, `-Dcohmetrix.url`).

## Index definitions and deviations

The indices follow Coh-Metrix 3.0 as published: Danielle S. McNamara, Arthur C.
Graesser, Philip M. McCarthy and Zhiqiang Cai,
[*Automated Evaluation of Text and Discourse with Coh-Metrix*](https://doi.org/10.1017/CBO9780511894664),
Cambridge University Press, 2014 ("the book"); Arthur C. Graesser, Danielle S.
McNamara, Max M. Louwerse and Zhiqiang Cai,
["Coh-Metrix: Analysis of text on cohesion and language"](https://doi.org/10.3758/BF03195564),
Behavior Research Methods, Instruments, & Computers 36(2), 193–202, 2004; and
Graesser, McNamara and Kulikowich (2011, see above). Where these sources leave a
point open, or where this component deviates from them, the definition is ours:

| Index | Definition here | Source, or why |
|---|---|---|
| WRDFRQc | mean base-10 logarithm of the content words' frequency per million | the book (ch. 4) describes the raw frequency; Graesser et al. (2004, p. 197) argue for the logarithm (reading time falls linearly with log frequency), and WRDFRQa and WRDFRQmc are logarithms in the book too |
| RDL2 | Coh-Metrix L2 Readability from the content word overlap of adjacent sentences (CRFCWO1), their syntactic similarity (SYNSTRUTa) and the word frequency as WRDFRQc (log mean over content words) | the frequency term deviates from the book's formula 5.1 (WRDFRQmc) and follows Scott A. Crossley, Jerry Greenfield and Danielle S. McNamara, ["Assessing Text Readability Using Cognitively Based Indices"](https://doi.org/10.1002/j.1545-7249.2008.tb00142.x), TESOL Quarterly 42(3), 475–493, 2008, and Scott A. Crossley, David B. Allen and Danielle S. McNamara, ["Text readability and intuitive simplification: A comparison of readability formulas"](https://scholarspace.manoa.hawaii.edu/items/fc754b04-a1ef-4176-885e-e4389a3a27e0/full), Reading in a Foreign Language 23(1), 84–101, 2011 (p. 91: the adjacent-sentence syntactic similarity) |
| SYNNP | per noun chunk the words that depend directly on the chunk's head, without punctuation; the mean over all chunks ("the lovely little girl" 3, "a very big dog" 2, "the old man's house" 1) | Graesser et al. (2004, p. 198): "A modifier is an optional element describing the property of the head of a phrase", "the lovely little girl" has three modifiers; the noun chunks and the rule beyond the example are ours |
| SYNLE | English: the words before the lexical verb of the main clause (in a copula clause the copula); German: before the finite verb | English as in Graesser, McNamara and Kulikowich (2011, p. 226: 16 words before "enrolled", the auxiliary "were" among them); German is our analogy, so that verb-final word order does not change what the index measures |
| LSAGN, LSAGNd | the first sentence enters the mean as 0 | without prior text the given part is 0, so G/(N+G) = 0 (the book, ch. 4). Each text contributes one 0, so the mean depends on the number of sentences (about 1/n); Maxim Konca measured rho .71 with the sentence count (.36 with the first sentence skipped), not verified by us |
| SMTEMP | tense and aspect repetition between adjacent sentences. German, following the TMV annotator (Anita Ramm, Sharid Loáiciga, Annemarie Friedrich and Alexander Fraser, ["Annotating tense, mood and voice for English, French and German"](https://aclanthology.org/P17-4001/), ACL 2017 System Demonstrations, Table 3): "würde" + infinitive future; Konjunktiv II the tense of its form (present; as a perfect, past); Perfekt completed. English: would/should/could/might conditional | the book names only tense (past, present, future) and aspect (in progress, completed); German has no conditional tense in TMV |
| DRGERUND | English: every -ing verb form (Penn tag VBG); not emitted for German | the book, ch. 4: "in its -ing form"; German has no gerund |
| WRDPOLc, WRDHYP* (German) | a lemma not in GermaNet is looked up in its old spelling, then as the head of a compound | our approximation (from Coh-Metrix 0.4.0-py) |
| DESWLsy (German) | the vowel-nucleus rule (see "Syllable counts") | from Coh-Metrix 0.4.0-py; its evaluation is Maxim Konca's, not verified by us |

Measurements of the reference preprocessing that this component relies on and
that are not verified by us: the German contraction table of `duui-stanza` was
built from UD German GSD and HDT, so its score on them (9 wrong tokens against
80 for Stanza's own splitting) is in-sample; how Stanza parses an unsplit "am"
before a superlative ("am besten") is not checked.

## Result semantics

`0.0` is emitted only when an actual calculation was possible and its result
is zero. If an index lacks required words, sentences, paragraphs, annotation
information, vectors, a valid comparison pair, a non-zero denominator, or a
required lexical resource, the final Python value is `None` with an `error`;
the Lua communication layer transfers it to the CAS as `Double.NaN`.

Some resource-backed indices use available-case evaluation: uncovered items
are excluded and a value is calculated from the remaining valid observations.
If too few valid observations remain, the result is `None`/`Double.NaN`.

Indices that are always `None` for a language are not emitted (label rule 9):
the Wiener Sachtextformel `RDWSTF1_ext` to `RDWSTF4_ext` for English (a German
formula) and `DRGERUND` for German. Records whose model waits for its refit are
emitted as `None` with the reason (the Text Easability components).

## Index labels

Every index record carries three labels. Follow the rules below when adding or
changing an index. The background and the decisions behind the scheme are in
[`docs/label-scheme.md`](docs/label-scheme.md).

| Field (Python / UIMA)        | Meaning                                                                                                                                                         | Unique per output |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| `label_v3` / `labelV3`       | **Public label.** The Coh-Metrix 3.0 label (Appendix A of Danielle S. McNamara, Arthur C. Graesser, Philip M. McCarthy and Zhiqiang Cai, [*Automated Evaluation of Text and Discourse with Coh-Metrix*](https://doi.org/10.1017/CBO9780511894664), Cambridge University Press, 2014), or `<NAME>_ext` for indices Coh-Metrix 3.0 does not have. Identical for all languages; the UIMA type. | Yes |
| `label_ttlab` / `labelTTLab` | **Internal label.** The public label plus groups naming the method and every resource the value depends on for this document, e.g. `WRDPOLc_GN`. May differ between languages and documents. | Yes |
| `label_v2` / `labelV2`       | Coh-Metrix 2.0 label from the same table, as a reference; `n/a` where the table has none, `None` for project-only indices.                                       | No                |

`INDEX_CATALOG` in `duui.py` lists every index with its public label, the
template of its internal label and its comparability (below); the `Index(...)`
calls take the internal label from it (`ttlab_labels`).

### Rules

1. **Public label.** The V3 label exactly as in Coh-Metrix 3.0. The only
   suffix is `_ext` (lowercase), for project-only indices: `index` 1001 and
   above, `label_v2` `None`. Example: `RDFOG_ext`.
2. **Internal label.** Always the public label plus zero or more groups:
   `<public>[_<GROUP>...]`. Indices without any resource have no groups
   (`DESWC`).
3. **Groups** are written like Coh-Metrix labels: an uppercase core with an
   optional lowercase variant (`SPACYlg`, `CLEARtr`, `LLMlist`), without
   underscores or hyphens. The method comes first (`REFIT`, `EMB`), then the
   data the value depends on, in input order (`SMCAUSvp_WN_LLMlist`: verbs,
   then particles).
4. **Name every resource**, including the original one (English `WN`,
   `MRC`): every lexicon, list, corpus, model or library the value depends
   on. Preprocessing (sentence boundaries, tokens, lemmas, POS, morphology,
   dependencies, noun chunks, Pyphen syllable counts) is not named; it is
   documented in the CAS metadata of the preprocessing components. Exception:
   dependency relations mapped to UD by this component (`TIGERmap`,
   `CLEARNLPmap`, see [Best effort without UD](#best-effort-without-ud)).
5. **No language codes.** Where the resource differs by language, its name
   says so (`WN`/`GN`, `CLEAR`/`CLEARtr`, `MRC`/`MRCtr`). Resources of the same
   kind (`LLMlist`, `WIKI`, `TEXTSTAT`) get the same name in every language;
   the document language is in the CAS. How a library handles each language
   is part of the library (see [textstat](#textstat)).
6. **No model names, no resource versions.** For spaCy vectors only the model
   size goes into the label (`SPACYsm`, `SPACYmd`, `SPACYlg`, `SPACYtrf`). A
   new version of a resource (SUBTLEX, Wikipedia sample, LLM list, MRC translation)
   bumps the index `version`, not the label. Exception: the name of the token
   vectors is not chosen by this component but by the tool or user that wrote
   them (`WordEmbedding.shortName`, see [Vectors](#vectors)).
7. **What was used at runtime.** If a fallback is used, the internal label
   names it. Example: without GermaNet, the German causal/intentional verb
   inventories use seed lists: `SMCAUSv_SEEDlist` instead of `SMCAUSv_GN`.
   Likewise for the resources supplied by the user: without DiMLex the
   connectives use LLM-generated lists (`CNCAll_LLMlist`), without SUBTLEX the
   Wikipedia samples (`WRDFRQc_WIKI`), without the native norms MRC
   (`WRDAOAc_MRC`). A fallback is computed only if the main record's resource
   is missing, and then replaces it (one record per index); with
   `all_records=true` it is also computed next to the main record.
8. **Derived indices** (RDL2, PC*) name the resources of their inputs
   (`RDL2_SUBTLEX`).
9. **No placeholders.** If the original resource is unavailable, emit only the
   substitute, under the original V3 label. Never emit a record that is always
   `None` just to occupy the original label. Example: there is no CELEX
   `WRDFRQc`; `WRDFRQc` is the SUBTLEX record (`WRDFRQc_SUBTLEX`). Exception:
   a record whose model waits for its refit is emitted as `None` with the
   reason in `error`, so its label stays stable (the Text Easability records,
   see [Labels and comparability](#labels-and-comparability)).
10. **UIMA type.** The CAS type is
    `org.texttechnologylab.uima.type.cohmetrix.<label_v3>`. A new index needs
    a matching type in `TypeSystem.xml`.

### Groups

| Group      | Kind   | Meaning                                                                                          |
|------------|--------|--------------------------------------------------------------------------------------------------|
| `REFIT`    | method | Model fitted again on another reference corpus (Text Easability PCA)                             |
| `EMB`      | method | Vector space instead of the TASA LSA space                                                       |
| `TIGERmap` | method | Dependency relations mapped from TIGER to UD (spaCy German, best effort)                         |
| `CLEARNLPmap` | method | Dependency relations mapped from ClearNLP to UD (spaCy English, best effort)                  |
| `REFa`     | data   | Reference corpus A (English): 9 registers × 1,000 passages of 250–350 words; reference corpus of the English Text Easability model |
| `REFv1`    | data   | Reference corpus v1 (German): 9 registers × 1,000 passages of 250–350 words; reference corpus of the German Text Easability model |
| `CLEAR`    | data   | CLEAR corpus, reference corpus of the earlier English Text Easability model                      |
| `CLEARtr`  | data   | CLEAR translated into German, reference corpus of the earlier German model                       |
| `SPACYxx`  | data   | spaCy token vectors from a model of size `xx` (`sm`, `md`, `lg`, `trf`); `SPACY` if unknown; other token vectors: their `shortName` (see [Vectors](#vectors)) |
| `DIMLEX`   | data   | DiMLex, lexicon of German discourse connectives                                                  |
| `DIMLEXeng` | data  | DiMLex-Eng, lexicon of English discourse connectives                                             |
| `GUM`      | data   | How often ambiguous English adverbs are connectives, counted on the GUM corpus (`gum_weights_en.csv`) |
| `LLMlist`  | data   | Word list generated with an LLM                                                                  |
| `WN`       | data   | Princeton WordNet                                                                                |
| `GN`       | data   | GermaNet                                                                                         |
| `SEEDlist` | data   | Hand-written seed list (fallback without GermaNet)                                               |
| `MRC`      | data   | MRC Psycholinguistic Database                                                                    |
| `MRCtr`    | data   | MRC translated into German                                                                       |
| `KUPERMAN` | data   | English age-of-acquisition norms (Kuperman et al.)                                               |
| `BIRCHENOUGH` | data | German age-of-acquisition norms (Birchenough et al.)                                            |
| `BRYSBAERT` | data  | English concreteness norms (Brysbaert et al.)                                                    |
| `KOEPER`   | data   | German concreteness and imageability norms (Köper & Schulte im Walde)                            |
| `SUBTLEX`  | data   | Word frequencies from film subtitles (SUBTLEX-US, SUBTLEX-DE) instead of CELEX                   |
| `WIKI`     | data   | Word frequencies from a Wikipedia sample instead of CELEX                                        |
| `TEXTSTAT` | data   | Computed with the textstat library                                                               |
| `TEXTSTATlist` | data | The easy-word list of the textstat library, in the formula of this component                   |

### Labels and comparability

The internal label names what was used, but not whether the value is
comparable to Coh-Metrix 3.0. That is the comparability column, per language
(also returned by `GET /v1/indices`):

| Value         | Meaning                                                                               |
|---------------|---------------------------------------------------------------------------------------|
| `original`    | Same method and resource as Coh-Metrix 3.0; differences come only from preprocessing |
| `counterpart` | The language's counterpart of the original resource or formula (GermaNet, Amstad)    |
| `translated`  | The original resource, translated                                                     |
| `corpus`      | Another corpus than the original (Wikipedia instead of CELEX)                         |
| `llmlist`     | A word list generated with an LLM instead of the original list                        |
| `emb`         | Another vector space than the TASA LSA space                                          |
| `refit`       | The model fitted again on another reference corpus                                    |
| `seedlist`    | A hand-written fallback list                                                          |
| `ext`         | Not in Coh-Metrix 3.0                                                                 |

German values are never directly comparable to English norms, also where the
column says `original`.

The main record of each index comes first. A **fallback** (rows "… fallback")
is computed only if a resource of the main record is missing, and then replaces
it; with `all_records=true` it is also computed next to the main record. A **variant** (another method or model: the textstat library, the CLEAR
models) is an additional record, selected like any internal label (e.g.
`exclude_indices=*_TEXTSTAT`). `GET /v1/indices` gives each record's `kind`
(`main`, `fallback`, `variant`). German
labels with `GN` need GermaNet; without it the verb inventories use the seed
lists (`SEEDlist`), and SMTEMP drops `GN`.

| Indices | Public | Internal EN | Internal DE | Comparable EN | Comparable DE |
|---|---|---|---|---|---|
| DES*, CRF*, LD*, SYN*, DR* except DRNEG, WRDNOUN … WRDPRP3p | V3 | same as public | same as public | original | original |
| PC* (16): models `refa-en-rr-gn-cr5` / `refv1-ud-rr-cr5` (computed since 0.7.0) | `PCNARz` | `PCNARz_REFIT_REFa` | `PCNARz_REFIT_REFv1` | refit | refit |
| … variant: models fitted on CLEAR / translated CLEAR (computed since 0.7.0) | | `PCNARz_REFIT_CLEAR` | `PCNARz_REFIT_CLEARtr` | refit | refit |
| LSA* (8), SMCAUSlsa | `LSASS1` | `LSASS1_EMB_SPACYlg` | `LSASS1_EMB_SPACYlg` | emb | emb |
| CNC* (9) | `CNCAll` | `CNCAll_DIMLEXeng_GUM` | `CNCAll_DIMLEX` | counterpart | counterpart |
| … fallback (without DiMLex / DiMLex-Eng) | | `CNCAll_LLMlist` | `CNCAll_LLMlist` | llmlist | llmlist |
| SMCAUSv, SMINTEp | `SMCAUSv` | `SMCAUSv_WN` | `SMCAUSv_GN` / `SMCAUSv_SEEDlist` | original | counterpart |
| SMCAUSvp, SMCAUSr | `SMCAUSvp` | `SMCAUSvp_WN_DIMLEXeng_GUM` | `SMCAUSvp_GN_DIMLEX` / `SMCAUSvp_SEEDlist_DIMLEX` | counterpart | counterpart |
| … fallback (without DiMLex / DiMLex-Eng) | | `SMCAUSvp_WN_LLMlist` | `SMCAUSvp_GN_LLMlist` / `SMCAUSvp_SEEDlist_LLMlist` | llmlist | counterpart, llmlist |
| SMINTEr | `SMINTEr` | `SMINTEr_WN_LLMlist` | `SMINTEr_GN_LLMlist` / `SMINTEr_SEEDlist_LLMlist` | original | counterpart |
| SMCAUSwn, WRDPOLc, WRDHYPn/v/nv | `WRDPOLc` | `WRDPOLc_WN` | `WRDPOLc_GN` | original | counterpart |
| SMTEMP | `SMTEMP` | `SMTEMP_LLMlist` | `SMTEMP_LLMlist_GN` / `SMTEMP_LLMlist` | llmlist | llmlist |
| DRNEG | `DRNEG` | `DRNEG_LLMlist` | `DRNEG_LLMlist` | llmlist | llmlist |
| WRDFRQc, WRDFRQa, WRDFRQmc | `WRDFRQc` | `WRDFRQc_SUBTLEX` | `WRDFRQc_SUBTLEX` | corpus | corpus |
| … fallback (without SUBTLEX) | | `WRDFRQc_WIKI` | `WRDFRQc_WIKI` | corpus | corpus |
| WRDAOAc | `WRDAOAc` | `WRDAOAc_KUPERMAN` | `WRDAOAc_BIRCHENOUGH` | counterpart | counterpart |
| … fallback (without the Kuperman / Birchenough norms) | | `WRDAOAc_MRC` | `WRDAOAc_MRCtr` | original | translated |
| WRDCNCc | `WRDCNCc` | `WRDCNCc_BRYSBAERT` | `WRDCNCc_KOEPER` | counterpart | counterpart |
| … fallback (without the Brysbaert / Köper norms) | | `WRDCNCc_MRC` | `WRDCNCc_MRCtr` | original | translated |
| WRDIMGc | `WRDIMGc` | `WRDIMGc_MRC` | `WRDIMGc_KOEPER` | original | counterpart |
| … fallback (German only, without the Köper norms) | | | `WRDIMGc_MRCtr` | | translated |
| WRDFAMc, WRDMEAc | `WRDFAMc` | `WRDFAMc_MRC` | `WRDFAMc_MRCtr` | original | translated |
| RDFRE | `RDFRE` | `RDFRE` | `RDFRE` | original | counterpart |
| RDFKGL | `RDFKGL` | `RDFKGL` | `RDFKGL` | original | original |
| RDSMOG, RDARI, RDCLI, RDLW, RDWSTF1–4 (all `_ext`) | `RDSMOG_ext` | `RDSMOG_ext` | `RDSMOG_ext` | ext | ext |
| RDFOG, RDDCRS, RDSPACHE (`_ext`; difficult words from textstat's easy-word list) | `RDFOG_ext` | `RDFOG_ext_TEXTSTATlist` | `RDFOG_ext_TEXTSTATlist` | ext | ext |
| … variant of every RD* above: the textstat library | | `RDFRE_TEXTSTAT`, `RDFOG_ext_TEXTSTAT` | `RDFRE_TEXTSTAT`, `RDFOG_ext_TEXTSTAT` | as the main record | as the main record |
| RDL2 | `RDL2` | `RDL2_SUBTLEX` | `RDL2_SUBTLEX` | corpus | corpus |
| … fallback (without SUBTLEX) | | `RDL2_WIKI` | `RDL2_WIKI` | corpus | corpus |

Rows with several indices show the first one; the others follow the same
pattern (`LSAGNd_EMB_SPACYlg`, `CNCCaus_DIMLEX`, `WRDHYPv_GN`, ...). Not
emitted (rule 9): DRGERUND for German (no gerund in German), RDWSTF1–4 for
English (a German formula). Other languages than German use the English
resources. The records that read the dependency relations get a mapping group
on spaCy-only input (see [Best effort without UD](#best-effort-without-ud)).

Adding a language changes only internal labels: a French wordnet keeps the
public label `WRDPOLc` and gets a new group, e.g. `WRDPOLc_WOLF`.

### Index version

Every record (an index, its fallback or a variant) has a `version` per language,
starting at `"1.0"` (`VERSIONS` in `duui.py`; `GET /v1/indices` lists
`version_de` and `version_en`). It versions the computation of that record,
independent of the image version in the annotation metadata, so results can be
compared record by record across releases. The languages can differ, e.g. in
0.6.0 German WRDIMGc uses new norms (`1.0`) while English WRDIMGc stays MRC
(`1.1`). Bump it in the same change that alters the value for the same input:

- minor (`1.0` -> `1.1`): values change, the meaning stays, e.g. a bug fix,
  changed `None`/edge-case handling, an updated word list or a new resource
  version (Wikipedia sample, MRC translation, spaCy model);
- major (`1.x` -> `2.0`): the meaning changes, e.g. a different formula,
  unit or resource type, usually together with a label change.

Label renames and refactorings or speedups with identical results do not
change the version.

## Validation

`CohMetrixUdPipelineTest` runs the default pipeline (see
[Prerequisites and preprocessing](#prerequisites-and-preprocessing)),
`CohMetrixBestEffortTest` the spaCy-only path, `CohMetrixInputLayersTest` the
choice of token and sentence layers; they need the running services (system
properties in the test classes).

The validation below was made for 0.2.0 (spaCy input) and does not apply to
0.6.0; a new set of frozen UD (Stanza) CASes is planned. The 0.2.0 Docker component was regression-tested through
`CohMetrixDockerValidationTest`. The Java test loads 322 frozen bilingual CAS
snapshots, removes previous Coh-Metrix output, processes each CAS with the
Docker image, and checks 3,496 curated index expectations. The frozen upstream
annotations make changes in Coh-Metrix, its Lua communication layer, and the
container service reproducible without rerunning spaCy or Syntok.

The final online and Maven-offline runs on 2026-09-17 used the same image
digest and produced:

```text
Tests run: 3499, Failures: 0, Errors: 0, Skipped: 9
BUILD SUCCESS
```

The nine skipped assertions are documented external limitations: two Pyphen
syllabification differences, three frozen spaCy annotation differences for
German `SMTEMP`, and four GermaNet expectations for which the licensed
resource is intentionally absent from the static test image.

Setup, execution commands, expectation semantics, and the remaining optional
test hardening are documented in
[`JAVA_VALIDATION_TESTS.md`](JAVA_VALIDATION_TESTS.md).

## How to use

For using duui-coh-metrix as a DUUI image it is necessary to use the
[Docker Unified UIMA Interface (DUUI)](https://github.com/texttechnologylab/DockerUnifiedUIMAInterface).

### Start the Docker container

```
docker run --rm -p 1000:9714 docker.texttechnologylab.org/v2/duui-coh-metrix:latest
```

Find all available image tags at
<https://docker.texttechnologylab.org/v2/duui-coh-metrix/tags/list>.

With GermaNet and the resources supplied by the user (see
[Resources supplied by the user](#resources-supplied-by-the-user)):

```
docker run --rm -p 1000:9714 \
  -v /path/to/germanet:/usr/src/app/src/main/resources/germanet \
  -v /path/to/external:/usr/src/app/src/main/resources/external \
  docker.texttechnologylab.org/v2/duui-coh-metrix:latest
```

### Run within DUUI

```java
DUUIComposer composer = new DUUIComposer()
    .withDebugLevel(DUUIComposer.DebugLevel.INFO)
    .withComponentLogging(true)
    .withDebugSeverity(true)
    .withDebugSource(true);

composer.add(
    new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
);
```

### Selecting indices

By default all indices are computed. Two optional parameters select a subset;
indices that are not selected are neither computed nor output:

```java
new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
    .withParameter("include_indices", "Descriptive,LSA")
    .withParameter("exclude_indices", "DESPC,LSAGN,LSAGNd")
```

Both take a comma-separated list of names, matched case-insensitively, from
least to most specific:

1. `*`: all indices;
2. an index group (the `type_name`, e.g. `Descriptive`, `LSA`, `Word Information`);
3. a prefix of the internal label, ending in `*`: `RDL2*` (RDL2 and
   its variants), `PCNAR*`, `LSAGN*` (LSAGN and LSAGNd), `WRDPOLc_G*`;
4. a public label (`LSAGN`, `RDFOG_ext`);
5. an internal label (`WRDPOLc_GN`, `LSASS1_EMB_SPACYlg`).

A name without `*` is exact: `LSAGN` does not select `LSAGNd`. Prefixes are
matched against the internal label, which starts with the public label, so
`WRDPOLc*` selects polysemy in every language, while `WRDPOLc_G*` or
`WRDPOLc_GN` only match German documents. Public labels and groups select the
same indices in a mixed-language corpus. `GET /v1/indices` lists all indices
with their group, public label, internal label per language (vectors as
`SPACY*`), comparability and version per language.

For each index, the most specific matching name decides; among prefixes, the
longer one. If the most specific match is in both lists, the index is
excluded. An empty `include_indices` means `*`. Examples:

| `include_indices` | `exclude_indices`      | Result                                  |
|-------------------|------------------------|-----------------------------------------|
|                   |                        | all indices                             |
| `Descriptive`     | `DESPC,DESSC`          | Descriptive without DESPC and DESSC     |
|                   | `LSA`                  | all indices except LSA                  |
| `DESWC,LSAGN`     |                        | only DESWC and LSAGN                    |
| `*,DESWC`         | `Descriptive`          | all except Descriptive, but with DESWC  |
| `DESWC`           | `Descriptive`          | only DESWC                              |
| `RD*`             | `RDL2*`                | Readability without RDL2                |
| `Readability`     | `*_ext`                | all Readability: `*` only works at the end, so `*_ext` is an unknown name (warning) |

Indices computed from other indices (`RDL2`, the Text Easability components)
compute their inputs as well, but the inputs are only output if they are
selected themselves. Unknown names are logged as a warning and ignored; for a
label of the previous scheme (e.g. `LSASS1_spacy`) the warning names the
public label to use. A name that only exists in another language (e.g.
`WRDPOLc_GN` in an English document) selects nothing, without a warning. If
`include_indices` contains only unknown names, nothing is computed.

Skipping indices also skips their shared preprocessing, e.g. the LSA vectors,
connective counts or hypernymy lookups.

**All records.** A fallback (e.g. `WRDFRQc_WIKI`, `CNCAll_LLMlist`,
`WRDAOAc_MRC`) is normally computed only if a resource of its index's main
record is missing, and then replaces it. With `all_records=true` the selected
indices also get their fallbacks next to the main record when its resources are
available, e.g. to show every value side by side:

```java
new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
    .withParameter("all_records", "true")
```

The selection applies as before (`include_indices=WRDFRQc` with
`all_records=true` gives `WRDFRQc_SUBTLEX` and `WRDFRQc_WIKI`). A missing resource
still makes the fallback replace its main record, and only then does the
fallback's description name the missing resource. Default `false`; other values
are refused (HTTP 400).

### Vectors

The LSA indices and `SMCAUSlsa` use token vectors. They are read from the
`WordEmbedding` annotations (`org.texttechnologylab.uima.type.WordEmbedding`,
one per token, same span as the token), e.g. written by `duui-spacy` (since
0.8.0) or any other tool. Each `WordEmbedding` has a `shortName`, the name of
its vector space, chosen by the tool or user that wrote it (`duui-spacy`:
`SPACY` + model size, e.g. `SPACYlg`, parameter `vectors_short_name`). The
optional parameter `vectors` selects them by this name:

```java
new DUUIDockerDriver.Component("docker.texttechnologylab.org/v2/duui-coh-metrix:latest")
    .withParameter("vectors", "SPACYlg")
```

- The name goes unchanged into the internal label (`LSASS1_EMB_SPACYlg`), so it
  must be an uppercase name with an optional lowercase variant
  (`[A-Z][A-Z0-9]*[a-z0-9]*`). This component needs no change for vectors of a
  new tool or model.
- Without the parameter, the only `shortName` in the CAS is used. If the CAS
  has vectors of several names, the LSA indices and `SMCAUSlsa` are `None`
  with an error that lists them; set `vectors` then.
- A name that is not in the CAS gives the error `No token vectors (<name>)`.

CASes without `WordEmbedding` annotations (e.g. from `duui-spacy` before
0.8.0, or with `vectors_in_spacytoken`) use the vectors of the `SpacyToken`
annotations, as before:

- `SPACY` (default): the model size goes into the internal label
  (`LSASS1_EMB_SPACYlg`) and is read from the CAS: the Lua layer sends the
  `modelName` of the `SpacyAnnotatorMetaData` that references the first
  `SpacyToken` (the layout of the `duui-spacy` component), and its last part
  (`sm`, `md`, `lg`, `trf`) is the size. Without that metadata the group is
  `SPACY` and a warning is logged. A size in the parameter (`SPACYlg`) is only
  used when the CAS has none; if the CAS says otherwise, the CAS wins and a
  warning is logged.
- Other names are not supported for `SpacyToken` vectors: the LSA indices are
  `None` with an error.

The size matters: a spaCy `sm` model has no static word vectors, so spaCy
returns rows of the contextual `doc.tensor` instead. A `trf` model returns no
token vectors at all. If no token of a document has a vector, the component
logs a warning and the LSA indices and `SMCAUSlsa` are `None` with the error
`No token vectors (<group>)`. Single tokens without a vector (words outside
the model's vocabulary) are normal and are skipped.

### Component logging

The Python service uses the official
[DUUIlogger](https://github.com/texttechnologylab/DUUIlogger). If component
logging is enabled in a logging-capable DUUI Core, request logs are returned
with the `/v1/process` response and are printed directly in the Java runner's
console. `INFO` shows request start/completion messages; `WARN` limits the
output to warnings and errors. `withDebugColorful(false)` is recommended for
CI logs and redirected files.

The transport is request/response based: Java receives the records after the
component response has arrived. Startup messages and logs from a process that
hangs or exits before sending its response remain available through Docker:

```bash
docker logs --timestamps --follow duui-coh-metrix-pipeline
```

## Cite

If you want to use the DUUI image please quote this as follows:

Alexander Leonhardt, Giuseppe Abrami, Daniel Baumartz and Alexander Mehler.
(2023). "Unlocking the Heterogeneous Landscape of Big Data NLP with DUUI."
Findings of the Association for Computational Linguistics: EMNLP 2023,
385–399. [[LINK](https://aclanthology.org/2023.findings-emnlp.29)]
[[PDF](https://aclanthology.org/2023.findings-emnlp.29.pdf)]

### BibTeX

```
@inproceedings{Leonhardt:et:al:2023,
  title     = {Unlocking the Heterogeneous Landscape of Big Data {NLP} with {DUUI}},
  author    = {Leonhardt, Alexander and Abrami, Giuseppe and Baumartz, Daniel and Mehler, Alexander},
  editor    = {Bouamor, Houda and Pino, Juan and Bali, Kalika},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2023.findings-emnlp.29},
  pages     = {385--399},
  pdf       = {https://aclanthology.org/2023.findings-emnlp.29.pdf},
  abstract  = {Automatic analysis of large corpora is a complex task, especially
               in terms of time efficiency. This complexity is increased by the
               fact that flexible, extensible text analysis requires the continuous
               integration of ever new tools. Since there are no adequate frameworks
               for these purposes in the field of NLP, and especially in the
               context of UIMA, that are not outdated or unusable for security
               reasons, we present a new approach to address the latter task:
               Docker Unified UIMA Interface (DUUI), a scalable, flexible, lightweight,
               and feature-rich framework for automatic distributed analysis
               of text corpora that leverages Big Data experience and virtualization
               with Docker. We evaluate DUUI{'}s communication approach against
               a state-of-the-art approach and demonstrate its outstanding behavior
               in terms of time efficiency, enabling the analysis of big text
               data.}
}

@misc{duui-coh-metrix,
  author         = {Baumartz, Daniel},
  title          = {Coh-Metrix Text Cohesion Analysis as {DUUI} component},
  year           = {2026},
  howpublished   = {https://github.com/texttechnologylab/duui-uima/tree/main/duui-coh-metrix}
}

```
