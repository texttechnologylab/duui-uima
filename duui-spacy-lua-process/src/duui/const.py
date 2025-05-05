from typing import Final, Literal

SpacyModelName = Literal[
    "ca_core_news_lg",
    "ca_core_news_md",
    "ca_core_news_sm",
    "ca_core_news_trf",
    "da_core_news_lg",
    "da_core_news_md",
    "da_core_news_sm",
    "da_core_news_trf",
    "de_core_news_lg",
    "de_core_news_md",
    "de_core_news_sm",
    "de_dep_news_trf",
    "el_core_news_lg",
    "el_core_news_md",
    "el_core_news_sm",
    "en_core_web_lg",
    "en_core_web_md",
    "en_core_web_sm",
    "en_core_web_trf",
    "es_core_news_lg",
    "es_core_news_md",
    "es_core_news_sm",
    "es_dep_news_trf",
    "fi_core_news_lg",
    "fi_core_news_md",
    "fi_core_news_sm",
    "fr_core_news_lg",
    "fr_core_news_md",
    "fr_core_news_sm",
    "fr_dep_news_trf",
    "hr_core_news_lg",
    "hr_core_news_md",
    "hr_core_news_sm",
    "it_core_news_lg",
    "it_core_news_md",
    "it_core_news_sm",
    "ja_core_news_lg",
    "ja_core_news_md",
    "ja_core_news_sm",
    "ja_core_news_trf",
    "ko_core_news_lg",
    "ko_core_news_md",
    "ko_core_news_sm",
    "lt_core_news_lg",
    "lt_core_news_md",
    "lt_core_news_sm",
    "mk_core_news_lg",
    "mk_core_news_md",
    "mk_core_news_sm",
    "nb_core_news_lg",
    "nb_core_news_md",
    "nb_core_news_sm",
    "nl_core_news_lg",
    "nl_core_news_md",
    "nl_core_news_sm",
    "pl_core_news_lg",
    "pl_core_news_md",
    "pl_core_news_sm",
    "pt_core_news_lg",
    "pt_core_news_md",
    "pt_core_news_sm",
    "ro_core_news_lg",
    "ro_core_news_md",
    "ro_core_news_sm",
    "ru_core_news_lg",
    "ru_core_news_md",
    "ru_core_news_sm",
    "sl_core_news_lg",
    "sl_core_news_md",
    "sl_core_news_sm",
    "sl_core_news_trf",
    "sv_core_news_lg",
    "sv_core_news_md",
    "sv_core_news_sm",
    "uk_core_news_lg",
    "uk_core_news_md",
    "uk_core_news_sm",
    "uk_core_news_trf",
    "xx_ent_wiki_sm",
    "xx_sent_ud_sm",
    "zh_core_web_lg",
    "zh_core_web_md",
    "zh_core_web_sm",
    "zh_core_web_trf",
]

SpacyModelSize = Literal["sm", "md", "lg", "trf", "efficiency", "accuracy"]

SpacyLanguage = Literal[
    "ca",
    "zh",
    "hr",
    "da",
    "nl",
    "en",
    "fi",
    "fr",
    "de",
    "el",
    "it",
    "ja",
    "ko",
    "lt",
    "mk",
    "nb",
    "pl",
    "pt",
    "ro",
    "ru",
    "sl",
    "es",
    "sv",
    "uk",
    "xx",
]

SPACY_LOOKUP: Final[dict[str, dict[str, SpacyModelName]]] = (
    {
        "efficiency": {
            "ca": "ca_core_news_sm",  # Catalan
            "zh": "zh_core_web_sm",  # Chinese
            "hr": "hr_core_news_sm",  # Croatian
            "da": "da_core_news_sm",  # Danish
            "nl": "nl_core_news_sm",  # Dutch
            "en": "en_core_web_sm",  # English
            "fi": "fi_core_news_sm",  # Finnish
            "fr": "fr_core_news_sm",  # French
            "de": "de_core_news_sm",  # German
            "el": "el_core_news_sm",  # Greek
            "it": "it_core_news_sm",  # Italian
            "ja": "ja_core_news_sm",  # Japanese
            "ko": "ko_core_news_sm",  # Korean
            "lt": "lt_core_news_sm",  # Lithuanian
            "mk": "mk_core_news_sm",  # Macedonian
            "nb": "nb_core_news_sm",  # Norwegian Bokmal
            "pl": "pl_core_news_sm",  # Polish
            "pt": "pt_core_news_sm",  # Portugese
            "ro": "ro_core_news_sm",  # Romanian
            "ru": "ru_core_news_sm",  # Russian
            "sl": "sl_core_news_sm",  # Slovenian
            "es": "es_core_news_sm",  # Spanish
            "sv": "sv_core_news_sm",  # Swedish
            "uk": "uk_core_news_sm",  # Ukrainian
            "xx": "xx_sent_ud_sm",  # Multi-Language / Unknown Language
        },
        "accuracy": {
            "ca": "ca_core_news_lg",
            "da": "da_core_news_lg",
            "de": "de_core_news_lg",
            "el": "el_core_news_lg",
            "en": "en_core_web_lg",
            "es": "es_core_news_lg",
            "fi": "fi_core_news_lg",
            "fr": "fr_core_news_lg",
            "hr": "hr_core_news_lg",
            "it": "it_core_news_lg",
            "ja": "ja_core_news_lg",
            "ko": "ko_core_news_lg",
            "lt": "lt_core_news_lg",
            "mk": "mk_core_news_lg",
            "nb": "nb_core_news_lg",
            "nl": "nl_core_news_lg",
            "pl": "pl_core_news_lg",
            "pt": "pt_core_news_lg",
            "ro": "ro_core_news_lg",
            "ru": "ru_core_news_lg",
            "sl": "sl_core_news_lg",
            "sv": "sv_core_news_lg",
            "uk": "uk_core_news_lg",
            "xx": "xx_ent_wiki_sm",
            "zh": "zh_core_web_lg",
        },
    }
    | {
        size: {
            lang: "_".join((model, size))
            for (lang, model) in [
                ("ca", "ca_core_news"),
                ("da", "da_core_news"),
                ("de", "de_core_news"),
                ("el", "el_core_news"),
                ("en", "en_core_web"),
                ("es", "es_core_news"),
                ("fi", "fi_core_news"),
                ("fr", "fr_core_news"),
                ("hr", "hr_core_news"),
                ("it", "it_core_news"),
                ("ja", "ja_core_news"),
                ("ko", "ko_core_news"),
                ("lt", "lt_core_news"),
                ("mk", "mk_core_news"),
                ("nb", "nb_core_news"),
                ("nl", "nl_core_news"),
                ("pl", "pl_core_news"),
                ("pt", "pt_core_news"),
                ("ro", "ro_core_news"),
                ("ru", "ru_core_news"),
                ("sl", "sl_core_news"),
                ("sv", "sv_core_news"),
                ("uk", "uk_core_news"),
                ("zh", "zh_core_web"),
            ]
        }
        for size in ["sm", "md", "lg"]
    }
    | {
        "trf": {
            "ca": "ca_core_news_trf",
            "da": "da_core_news_trf",
            "de": "de_dep_news_trf",
            "en": "en_core_web_trf",
            "es": "es_dep_news_trf",
            "fr": "fr_dep_news_trf",
            "ja": "ja_core_news_trf",
            "sl": "sl_core_news_trf",
            "uk": "uk_core_news_trf",
            "zh": "zh_core_web_trf",
        }
    }
)


SpacyPipelineComponent = Literal[
    "tagger",
    "parser",
    "ner",
    "entity_linker",
    "entity_ruler",
    "textcat",
    "textcat_multilabel",
    "lemmatizer",
    "trainable_lemmatizer",
    "morphologizer",
    "attribute_ruler",
    "senter",
    "sentencizer",
    "tok2vec",
    "transformer",
]


with open("communication_layer.lua", "r") as f:
    LUA_COMMUNICATION_LAYER: Final[str] = f.read()
