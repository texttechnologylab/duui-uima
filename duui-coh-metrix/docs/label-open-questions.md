# Coh-Metrix labels: open questions

Context: every index has

- `label_v3` / `label_v2`: the original Coh-Metrix label (concept identity),
- a public TTLab label (`label_ttlab`) that gets a suffix whenever the English
  implementation deviates from the original in *meaning* (different corpus,
  lexicon, word list, model, formula or library), but not for preprocessing
  differences (tokenization, sentence splitting, POS, parsing, lemmas,
  syllables),
- an internal TTLab label (`label_internal`) that names the concrete resource
  per language.

Test for a suffix: *If our NLP preprocessing were perfect, would we reproduce
the original value?* If not, the label needs a suffix that names the reason.
The full rules are in `README.md` under "Index labels".

The items below were checked against the implementation documentation
(`docs/Coh_Metrix Voll.pdf`, page numbers refer to it). Line numbers refer to
`src/main/python/duui.py` and may shift.

## Resolved

- **Connectives (`CNC*_llmlist`)**: "Liste generiert mit ChatGPT 4o" for the
  language-specific inventories (PDF, section 8; `duui.py:2668`). The suffix
  `_gptlist` was renamed to `_llmlist` (suffixes name the kind of resource,
  not the model).
- **`SMTEMP_llmlist`**: "Die hier verwendeten Wortlisten wurden mit GPT-5.6
  Sol erzeugt" (p. 86). The tense/aspect detection uses word lists
  (`_DE_COMPLETION_*`, `_EN_FUTURE_MODALS`, `_EN_CONDITIONAL_MODALS`), so the
  suffix is right.
- **`SMINTEr_llmlist`**: the intentional particle lists were generated with
  GPT-5.6 Sol (p. 82). Renamed from `SMINTEr_wordnet`; internal label for
  German `SMINTEr_germanet_llmlist` (or `_seedlist_llmlist` without
  GermaNet).
- **`RDFRE_textstat`**: for German, textstat uses the Amstad variant (p. 141).
  No extra suffix: the label names the library, not the formula it uses
  internally (README rule 10).

## Still open

1. **`DRNEG_llmlist`**: the PDF only says "konservative sprachspezifische
   Lemmaliste" (p. 103). Were `_DE_NEGATION_LEMMAS` / `_EN_NEGATION_LEMMAS`
   generated with an LLM? If they are hand-written, the label would be
   `DRNEG` (or another suffix).
2. **Causal particles (`SMCAUSvp`, `SMCAUSr`)**: the particle lists are only
   called "projektspezifische Approximationen" (p. 78). Generated with an LLM
   like the intentional particles? Then `SMCAUSvp` and `SMCAUSr` become
   `_llmlist` as well.
3. **Verb classes (`SMCAUSv`, `SMCAUSvp`, `SMINTEp`, `SMCAUSr`; currently
   `_wordnet`)**: English uses `cause.v.01` and `intend.v.01` + `plan.v.01`
   with hyponyms, German expands seeds via GermaNet with the seed lists as
   fallback (pp. 76, 79). Who chose the synsets and seeds? The original
   Coh-Metrix also classifies verbs with WordNet, so if the selection is
   considered an implementation detail, the public labels lose the suffix
   (`SMCAUSv`, `SMINTEp`).
4. **`RDFKGL_textstat`**: does the textstat fork change anything for German,
   and which syllable counter does it use? This does not change the label
   (rule 10), but should be documented.
