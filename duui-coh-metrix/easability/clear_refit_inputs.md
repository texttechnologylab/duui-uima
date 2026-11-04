# CLEAR refit: which index records go into the PCA

Generated 2026-11-04 from `easability/pca_input_measures.csv` and the index catalog of `src/main/python/duui.py` (duui-coh-metrix 0.6.0). The PCA takes the **main record** of each input index, by internal label and language (columns `labels_en`, `labels_de` of `pca_input_measures.csv`; stored in the model as `labels_internal`). German labels with `GN` assume GermaNet is mounted. Measure numbers and names: Arthur C. Graesser, Danielle S. McNamara and Jonna M. Kulikowich, ["Coh-Metrix: Providing Multilevel Analyses of Text Characteristics"](https://doi.org/10.3102/0013189X11413260), Educational Researcher 40(5), 223–234, 2011, Table 1.

## Inputs

| # | Measure (2011) | English | German | Note |
|---|---|---|---|---|
| 1 | Syllables per word | `DESWLsy` | `DESWLsy` |  |
| 2 | Nouns | `WRDNOUN` | `WRDNOUN` |  |
| 3 | Verbs | `WRDVERB` | `WRDVERB` |  |
| 4 | Adjectives | `WRDADJ` | `WRDADJ` |  |
| 5 | Adverbs | `WRDADV` | `WRDADV` |  |
| 6 | Pronouns | `WRDPRO` | `WRDPRO` |  |
| 7 | First-person pronouns | `WRDPRP1s` + `WRDPRP1p` | `WRDPRP1s` + `WRDPRP1p` | sum of singular and plural incidence |
| 8 | Third-person pronouns | `WRDPRP3s` + `WRDPRP3p` | `WRDPRP3s` + `WRDPRP3p` | sum of singular and plural incidence |
| 10 | Connectives | `CNCAll_DIMLEXeng_GUM` | `CNCAll_DIMLEX` |  |
| 11 | Causal connectives | `CNCCaus_DIMLEXeng_GUM` | `CNCCaus_DIMLEX` |  |
| 12 | Temporal connectives | `CNCTemp_DIMLEXeng_GUM` | `CNCTemp_DIMLEX` |  |
| 13 | Logical connectives | `CNCLogic_DIMLEXeng_GUM` | `CNCLogic_DIMLEX` |  |
| 14 | Additive connectives | `CNCAdd_DIMLEXeng_GUM` | `CNCAdd_DIMLEX` |  |
| 15 | Adversative connectives | `CNCADC_DIMLEXeng_GUM` | `CNCADC_DIMLEX` |  |
| 16 | Word frequency (logarithm) | `WRDFRQa_SUBTLEX` | `WRDFRQa_SUBTLEX` | SUBTLEX instead of CELEX |
| 17 | Content word frequency (logarithm) | `WRDFRQc_SUBTLEX` | `WRDFRQc_SUBTLEX` | SUBTLEX instead of CELEX; 3.0 index is not logarithmic |
| 18 | Minimum word frequency per sentence | `WRDFRQmc_SUBTLEX` | `WRDFRQmc_SUBTLEX` | SUBTLEX instead of CELEX; content words only in 3.0 |
| 19 | Age of acquisition | `WRDAOAc_KUPERMAN` | `WRDAOAc_BIRCHENOUGH` |  |
| 20 | Meaningfulness | `WRDMEAc_MRC` | `WRDMEAc_MRCtr` |  |
| 21 | Concreteness | `WRDCNCc_BRYSBAERT` | `WRDCNCc_KOEPER` |  |
| 22 | Imagery | `WRDIMGc_MRC` | `WRDIMGc_KOEPER` |  |
| 23 | Familiarity | `WRDFAMc_MRC` | `WRDFAMc_MRCtr` |  |
| 24 | Negations | `DRNEG_LLMlist` | `DRNEG_LLMlist` |  |
| 25 | Causal verbs | `SMCAUSv_WN` | `SMCAUSv_GN` |  |
| 26 | Intentional actions, events, and particles | `SMINTEp_WN` | `SMINTEp_GN` |  |
| 27 | Polysemy—multiple senses of word | `WRDPOLc_WN` | `WRDPOLc_GN` |  |
| 28 | Words per sentence | `DESSL` | `DESSL` |  |
| 29 | Modifiers per noun phrase | `SYNNP` | `SYNNP` |  |
| 30 | Words before main verb of main clause | `SYNLE` | `SYNLE` |  |
| 31 | Passive constructions | `DRPVAL` | `DRPVAL` |  |
| 32 | Syntactic similarity—sentences in paragraph | `SYNSTRUTa` | `SYNSTRUTa` | 3.0 has adjacent sentences (SYNSTRUTa) and all sentences across paragraphs (SYNSTRUTt); SYNSTRUTt is undefined for single-paragraph texts |
| 33 | Content word overlap—adjacent sentences | `CRFCWO1` | `CRFCWO1` |  |
| 34 | Content word overlap—all sentences | `CRFCWOa` | `CRFCWOa` |  |
| 35 | Argument overlap—adjacent sentences | `CRFAO1` | `CRFAO1` |  |
| 36 | Argument overlap—all sentences | `CRFAOa` | `CRFAOa` |  |
| 37 | Noun overlap—adjacent sentences | `CRFNO1` | `CRFNO1` |  |
| 38 | Stem overlap—all sentences | `CRFSOa` | `CRFSOa` |  |
| 39 | Type-token ratio | `LDTTRc` | `LDTTRc` | 3.0 has content words (LDTTRc) and all words (LDTTRa) |
| 40 | Lexical diversity—all words | `LDMTLDa` | `LDMTLDa` | 3.0 has MTLD (LDMTLDa) and VOCD (LDVOCDa) |
| 42 | LSA-givenness versus newness | `LSAGN_EMB_SPACYlg` | `LSAGN_EMB_SPACYlg` | spaCy lg vectors instead of TASA LSA space |
| 43 | LSA overlap—adjacent sentences | `LSASS1_EMB_SPACYlg` | `LSASS1_EMB_SPACYlg` | spaCy lg vectors instead of TASA LSA space |
| 44 | LSA overlap—all sentences | `LSASSp_EMB_SPACYlg` | `LSASSp_EMB_SPACYlg` | spaCy lg vectors instead of TASA LSA space; 3.0 is all sentences in paragraph |
| 45 | Dissimilarity of parts of speech between sentences | `SYNMEDpos` | `SYNMEDpos` |  |
| 46 | Dissimilarity of words between sentences | `SYNMEDwrd` | `SYNMEDwrd` |  |
| 47 | Causal cohesion | `SMCAUSr_WN_DIMLEXeng_GUM` | `SMCAUSr_GN_DIMLEX` |  |
| 48 | Intentional cohesion | `SMINTEr_WN_LLMlist` | `SMINTEr_GN_LLMlist` | excluded until 0.6.0 (undefined for 71% of CLEAR with 0.2.0's particles / verbs); since 0.6.0 particles / (verbs + 1), always defined (decided 2026-11-04) |
| 49 | Verb overlap—adjacent | `SMCAUSwn_WN` | `SMCAUSwn_GN` |  |
| 50 | LSA verb overlap—adjacent sentences | `SMCAUSlsa_EMB_SPACYlg` | `SMCAUSlsa_EMB_SPACYlg` | spaCy lg vectors instead of TASA LSA space |
| 51 | Temporal cohesion | `SMTEMP_LLMlist` | `SMTEMP_LLMlist_GN` |  |

49 measures from 51 records per language (measures 7 and 8 are the sum of two records each).

## Measures of the 2011 table that are not inputs

| # | Measure (2011) | Why |
|---|---|---|
| 9 | Ratio of function words to content words | no index: not implemented in Coh-Metrix 3.0 |
| 41 | Lexical diversity—verbs | no index: not implemented in Coh-Metrix 3.0 |
| 52 | Verb tense repetition | no index: 3.0 combines tense and aspect repetition in SMTEMP (measure 51) |
| 53 | Verb aspect repetition | no index: 3.0 combines tense and aspect repetition in SMTEMP (measure 51) |

## Index records that are not inputs

Every other record the component computes for the language. The Text Easability components are not computed for the reference corpus (`exclude_indices`).

### English: 155 records, 51 inputs, 104 not inputs

- **main records of indices that are not among the 2011 measures** (39): `DESPC`, `DESSC`, `DESWC`, `DESPL`, `DESPLd`, `DESSLd`, `DESWLsyd`, `DESWLlt`, `DESWLltd`, `CRFSO1`, `CRFNOa`, `CRFCWO1d`, `CRFCWOad`, `LSASS1d_EMB_SPACYlg`, `LSASSpd_EMB_SPACYlg`, `LSAPP1_EMB_SPACYlg`, `LSAPP1d_EMB_SPACYlg`, `LSAGNd_EMB_SPACYlg`, `LDTTRa`, `LDVOCDa`, `CNCTempx_DIMLEXeng_GUM`, `CNCPos_DIMLEXeng_GUM`, `CNCNeg_DIMLEXeng_GUM`, `SMCAUSvp_WN_DIMLEXeng_GUM`, `SYNMEDlem`, `SYNSTRUTt`, `DRNP`, `DRVP`, `DRAP`, `DRPP`, `DRGERUND`, `DRINF`, `WRDPRP2`, `WRDHYPn_WN`, `WRDHYPv_WN`, `WRDHYPnv_WN`, `RDFRE`, `RDFKGL`, `RDL2_SUBTLEX`
- **Text Easability components (the output; main models)** (16): `PCNARz_REFIT_REFa`, `PCNARp_REFIT_REFa`, `PCSYNz_REFIT_REFa`, `PCSYNp_REFIT_REFa`, `PCCNCz_REFIT_REFa`, `PCCNCp_REFIT_REFa`, `PCREFz_REFIT_REFa`, `PCREFp_REFIT_REFa`, `PCDCz_REFIT_REFa`, `PCDCp_REFIT_REFa`, `PCVERBz_REFIT_REFa`, `PCVERBp_REFIT_REFa`, `PCCONNz_REFIT_REFa`, `PCCONNp_REFIT_REFa`, `PCTEMPz_REFIT_REFa`, `PCTEMPp_REFIT_REFa`
- **Text Easability components (the output; CLEAR models)** (16): `PCNARz_REFIT_CLEAR`, `PCNARp_REFIT_CLEAR`, `PCSYNz_REFIT_CLEAR`, `PCSYNp_REFIT_CLEAR`, `PCCNCz_REFIT_CLEAR`, `PCCNCp_REFIT_CLEAR`, `PCREFz_REFIT_CLEAR`, `PCREFp_REFIT_CLEAR`, `PCDCz_REFIT_CLEAR`, `PCDCp_REFIT_CLEAR`, `PCVERBz_REFIT_CLEAR`, `PCVERBp_REFIT_CLEAR`, `PCCONNz_REFIT_CLEAR`, `PCCONNp_REFIT_CLEAR`, `PCTEMPz_REFIT_CLEAR`, `PCTEMPp_REFIT_CLEAR`
- **variants (other resource or method; the main record is the input or the index is no input)** (26): `CNCAll_LLMlist`, `CNCCaus_LLMlist`, `CNCLogic_LLMlist`, `CNCADC_LLMlist`, `CNCTemp_LLMlist`, `CNCTempx_LLMlist`, `CNCAdd_LLMlist`, `CNCPos_LLMlist`, `CNCNeg_LLMlist`, `SMCAUSvp_WN_LLMlist`, `SMCAUSr_WN_LLMlist`, `WRDFRQc_WIKI`, `WRDFRQa_WIKI`, `WRDFRQmc_WIKI`, `WRDAOAc_MRC`, `WRDCNCc_MRC`, `RDFRE_TEXTSTAT`, `RDFKGL_TEXTSTAT`, `RDFOG_ext_TEXTSTAT`, `RDSMOG_ext_TEXTSTAT`, `RDARI_ext_TEXTSTAT`, `RDCLI_ext_TEXTSTAT`, `RDLW_ext_TEXTSTAT`, `RDDCRS_ext_TEXTSTAT`, `RDSPACHE_ext_TEXTSTAT`, `RDL2_WIKI`
- **project-only readability formulas (`_ext`), not in the 2011 measures** (7): `RDFOG_ext_TEXTSTATlist`, `RDSMOG_ext`, `RDARI_ext`, `RDCLI_ext`, `RDLW_ext`, `RDDCRS_ext_TEXTSTATlist`, `RDSPACHE_ext_TEXTSTATlist`

### German: 163 records, 51 inputs, 112 not inputs

- **main records of indices that are not among the 2011 measures** (38): `DESPC`, `DESSC`, `DESWC`, `DESPL`, `DESPLd`, `DESSLd`, `DESWLsyd`, `DESWLlt`, `DESWLltd`, `CRFSO1`, `CRFNOa`, `CRFCWO1d`, `CRFCWOad`, `LSASS1d_EMB_SPACYlg`, `LSASSpd_EMB_SPACYlg`, `LSAPP1_EMB_SPACYlg`, `LSAPP1d_EMB_SPACYlg`, `LSAGNd_EMB_SPACYlg`, `LDTTRa`, `LDVOCDa`, `CNCTempx_DIMLEX`, `CNCPos_DIMLEX`, `CNCNeg_DIMLEX`, `SMCAUSvp_GN_DIMLEX`, `SYNMEDlem`, `SYNSTRUTt`, `DRNP`, `DRVP`, `DRAP`, `DRPP`, `DRINF`, `WRDPRP2`, `WRDHYPn_GN`, `WRDHYPv_GN`, `WRDHYPnv_GN`, `RDFRE`, `RDFKGL`, `RDL2_SUBTLEX`
- **Text Easability components (the output; main models)** (16): `PCNARz_REFIT_REFv1`, `PCNARp_REFIT_REFv1`, `PCSYNz_REFIT_REFv1`, `PCSYNp_REFIT_REFv1`, `PCCNCz_REFIT_REFv1`, `PCCNCp_REFIT_REFv1`, `PCREFz_REFIT_REFv1`, `PCREFp_REFIT_REFv1`, `PCDCz_REFIT_REFv1`, `PCDCp_REFIT_REFv1`, `PCVERBz_REFIT_REFv1`, `PCVERBp_REFIT_REFv1`, `PCCONNz_REFIT_REFv1`, `PCCONNp_REFIT_REFv1`, `PCTEMPz_REFIT_REFv1`, `PCTEMPp_REFIT_REFv1`
- **Text Easability components (the output; CLEAR models)** (16): `PCNARz_REFIT_CLEARtr`, `PCNARp_REFIT_CLEARtr`, `PCSYNz_REFIT_CLEARtr`, `PCSYNp_REFIT_CLEARtr`, `PCCNCz_REFIT_CLEARtr`, `PCCNCp_REFIT_CLEARtr`, `PCREFz_REFIT_CLEARtr`, `PCREFp_REFIT_CLEARtr`, `PCDCz_REFIT_CLEARtr`, `PCDCp_REFIT_CLEARtr`, `PCVERBz_REFIT_CLEARtr`, `PCVERBp_REFIT_CLEARtr`, `PCCONNz_REFIT_CLEARtr`, `PCCONNp_REFIT_CLEARtr`, `PCTEMPz_REFIT_CLEARtr`, `PCTEMPp_REFIT_CLEARtr`
- **variants (other resource or method; the main record is the input or the index is no input)** (31): `CNCAll_LLMlist`, `CNCCaus_LLMlist`, `CNCLogic_LLMlist`, `CNCADC_LLMlist`, `CNCTemp_LLMlist`, `CNCTempx_LLMlist`, `CNCAdd_LLMlist`, `CNCPos_LLMlist`, `CNCNeg_LLMlist`, `SMCAUSvp_GN_LLMlist`, `SMCAUSr_GN_LLMlist`, `WRDFRQc_WIKI`, `WRDFRQa_WIKI`, `WRDFRQmc_WIKI`, `WRDAOAc_MRCtr`, `WRDCNCc_MRCtr`, `WRDIMGc_MRCtr`, `RDFRE_TEXTSTAT`, `RDFKGL_TEXTSTAT`, `RDFOG_ext_TEXTSTAT`, `RDSMOG_ext_TEXTSTAT`, `RDARI_ext_TEXTSTAT`, `RDCLI_ext_TEXTSTAT`, `RDLW_ext_TEXTSTAT`, `RDDCRS_ext_TEXTSTAT`, `RDSPACHE_ext_TEXTSTAT`, `RDWSTF1_ext_TEXTSTAT`, `RDWSTF2_ext_TEXTSTAT`, `RDWSTF3_ext_TEXTSTAT`, `RDWSTF4_ext_TEXTSTAT`, `RDL2_WIKI`
- **project-only readability formulas (`_ext`), not in the 2011 measures** (11): `RDFOG_ext_TEXTSTATlist`, `RDSMOG_ext`, `RDARI_ext`, `RDCLI_ext`, `RDLW_ext`, `RDDCRS_ext_TEXTSTATlist`, `RDSPACHE_ext_TEXTSTATlist`, `RDWSTF1_ext`, `RDWSTF2_ext`, `RDWSTF3_ext`, `RDWSTF4_ext`

## Notes

- The reference corpus must be made with GermaNet mounted: without it the German verb records are `_SEEDlist` instead of `_GN`, and `pca.py` stops because the input columns are missing.
- The inputs need the resources of the external volume (README, "Resources"): the frequencies SUBTLEX (English SUBTLEX-US; German SUBTLEX-DE, see below), the norms, DiMLex / DiMLex-Eng.
- **SUBTLEX-DE is required for the German run** (measures 16–18): `download_resources.py` takes the authors' release (all 190,500 forms, from the Internet Archive; README "Resources supplied by the user"), not CTAP's subset of 65,499 forms; with CTAP's table the German frequency inputs come out higher.
- Measure 48 (SMINTEr) was excluded until 0.6.0 (undefined for 71 % of CLEAR with 0.2.0's particles / verbs). Since 0.6.0 SMINTEr is particles / (verbs + 1) and always defined; included again (decided 2026-11-04).
- The REFa/REFv1 models of Coh-Metrix 0.5.0-py use another selection (44 or 45 inputs: one of each pair with |r| > .90 removed); the CLEAR models keep the 2011 measures unchanged.
