from __future__ import annotations

from typing import Any, Optional, Union

import coreferee  # noqa: F401  # Registers the spaCy pipeline component "coreferee"
import spacy

from spacy.tokens import Doc


EXTERNAL_OFFSETS_EXTENSION = "external_token_offsets"

if not Doc.has_extension(EXTERNAL_OFFSETS_EXTENSION):
    Doc.set_extension(EXTERNAL_OFFSETS_EXTENSION, default=None)


class CorefereeResolver:
    """
    Coreferee wrapper with one fixed language per instance.

    Input:
        - tokens: list[str]
        - begins: list[int]
        - ends: list[int]

    Output:
        {
            "begin": [...],
            "end": [...],
            "begin_resolve": [...],
            "end_resolve": [...],
            "token": [...],
            "token_resolve": [...],
        }

    The language is set once during initialization.
    Runtime language switching is intentionally not supported.
    """

    DEFAULT_MODELS = {
        "sm": {
            "en": "en_core_web_sm",
            "de": "de_core_news_sm",
            "fr": "fr_core_news_sm",
            "pl": "pl_core_news_md",
        },
        "lg": {
            "en": "en_core_web_lg",
            "de": "de_core_news_lg",
            "fr": "fr_core_news_lg",
            "pl": "pl_core_news_lg",
        },
    }

    LANG_ALIASES = {
        "en": "en",
        "english": "en",
        "englisch": "en",

        "de": "de",
        "german": "de",
        "deutsch": "de",

        "fr": "fr",
        "french": "fr",
        "französisch": "fr",
        "franzoesisch": "fr",

        "pl": "pl",
        "polish": "pl",
        "polnisch": "pl",
    }

    def __init__(
            self,
            language: str,
            variant: str,
            model_overrides: Optional[dict[str, str]] = None,
    ):
        self._language = self._normalize_language(language)
        self.variant = variant

        self.models = dict(self.DEFAULT_MODELS)
        if model_overrides:
            self.models.update(model_overrides)

        self.nlp = self._load_pipeline()

    @property
    def language(self) -> str:
        return self._language

    def _normalize_language(self, language: str) -> str:
        lang = language.strip().lower()

        if lang not in self.LANG_ALIASES:
            supported = ", ".join(self.DEFAULT_MODELS.keys())
            raise ValueError(
                f"Unsupported language: {language!r}. "
                f"Supported languages are: {supported}"
            )

        return self.LANG_ALIASES[lang]

    def _load_pipeline(self):
        model_name = self.models[self.variant][self.language]

        try:
            nlp = spacy.load(model_name)
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model not found: {model_name!r}\n\n"
                f"Install it with:\n"
                f"python -m spacy download {model_name}\n\n"
                f"Or override the model name, for example:\n"
                f"CorefereeResolver('en', model_overrides={{'en': 'en_core_web_sm'}})"
            ) from exc

        if "coreferee" not in nlp.pipe_names:
            try:
                nlp.add_pipe("coreferee")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load Coreferee for language {self.language!r}.\n\n"
                    f"Install the Coreferee language data with:\n"
                    f"python -m coreferee install {self.language}"
                ) from exc

        return nlp

    def process_text(self, text: str) -> Doc:
        """
        Process raw text.

        This is optional. If you already have tokens/begins/ends,
        use process_tokens instead.

        For raw text, external offsets are taken from spaCy token offsets.
        """
        if not text or not text.strip():
            raise ValueError("text must not be empty.")

        doc = self.nlp(text)

        doc._.external_token_offsets = [
            {
                "begin": token.idx,
                "end": token.idx + len(token.text),
            }
            for token in doc
        ]

        return doc

    def process_tokens(
            self,
            tokens: list[str],
            begins: list[int],
            ends: list[int],
            spaces: Optional[list[bool]] = None,
    ) -> Doc:
        """
        Process pre-tokenized input with external begin/end offsets.

        Args:
            tokens:
                Separate list of token strings.

            begins:
                Separate list of begin offsets.

            ends:
                Separate list of end offsets.

            spaces:
                Optional whitespace information.
                If None, spaces are inferred from begin/end offsets.

        Important:
            This method does not call self.nlp(" ".join(tokens)),
            because that would let spaCy tokenize the text again.
        """
        self._validate_token_offsets(tokens, begins, ends)

        if spaces is None:
            spaces = self._infer_spaces_from_offsets(begins, ends)

        if len(tokens) != len(spaces):
            raise ValueError(
                "tokens and spaces must have the same length. "
                f"tokens={len(tokens)}, spaces={len(spaces)}"
            )

        doc = Doc(self.nlp.vocab, words=tokens, spaces=spaces)

        doc._.external_token_offsets = [
            {
                "begin": int(begin),
                "end": int(end),
            }
            for begin, end in zip(begins, ends)
        ]

        for _, component in self.nlp.pipeline:
            doc = component(doc)

        return doc

    def process(
            self,
            input_data: Union[str, list[str]],
            begins: Optional[list[int]] = None,
            ends: Optional[list[int]] = None,
            spaces: Optional[list[bool]] = None,
    ) -> Doc:
        """
        Generic input processor.

        Supported:
            - str
            - list[str] with begins and ends

        If input_data is list[str], begins and ends are required.
        """
        if isinstance(input_data, str):
            return self.process_text(input_data)

        if isinstance(input_data, list):
            if begins is None or ends is None:
                raise ValueError(
                    "begins and ends are required when input_data is a token list."
                )

            return self.process_tokens(
                tokens=input_data,
                begins=begins,
                ends=ends,
                spaces=spaces,
            )

        raise TypeError("input_data must be either a string or a list of tokens.")

    def get_coreference_dict(
            self,
            doc: Doc,
            include_self: bool = False,
            expand_noun_chunks: bool = True,
    ) -> dict[str, list]:
        """
        Return all detected coreferences as a dictionary with six lists.

        Output:
            {
                "begin": [...],
                "end": [...],
                "begin_resolve": [...],
                "end_resolve": [...],
                "token": [...],
                "token_resolve": [...],
            }

        Meaning:
            begin[i], end[i], token[i]
                The detected mention.

            begin_resolve[i], end_resolve[i], token_resolve[i]
                The resolved mention of the same coreference chain.

        No pronoun list is required.
        """
        if doc._.external_token_offsets is None:
            raise RuntimeError(
                "The Doc has no external offsets. "
                "Use process_tokens(tokens, begins, ends) or process_text(text)."
            )

        result: dict[str, list] = {
            "begin": [],
            "end": [],
            "begin_resolve": [],
            "end_resolve": [],
            "token": [],
            "token_resolve": [],
        }

        seen: set[tuple[int, int, int, int]] = set()

        for chain in doc._.coref_chains:
            mentions = self._get_chain_mentions(chain)

            if not mentions:
                continue

            representative_index = getattr(
                chain,
                "most_specific_mention_index",
                0,
            )

            if representative_index is None:
                representative_index = 0

            if representative_index < 0 or representative_index >= len(mentions):
                representative_index = 0

            representative_mention = mentions[representative_index]

            resolved_span = self._mention_to_external_span(
                doc=doc,
                mention=representative_mention,
                expand_noun_chunks=expand_noun_chunks,
            )

            for mention in mentions:
                mention_span = self._mention_to_external_span(
                    doc=doc,
                    mention=mention,
                    expand_noun_chunks=expand_noun_chunks,
                )

                same_span = (
                        mention_span["begin"] == resolved_span["begin"]
                        and mention_span["end"] == resolved_span["end"]
                )

                if same_span and not include_self:
                    continue

                key = (
                    mention_span["begin"],
                    mention_span["end"],
                    resolved_span["begin"],
                    resolved_span["end"],
                )

                if key in seen:
                    continue

                seen.add(key)

                result["begin"].append(mention_span["begin"])
                result["end"].append(mention_span["end"])
                result["begin_resolve"].append(resolved_span["begin"])
                result["end_resolve"].append(resolved_span["end"])
                result["token"].append(mention_span["text"])
                result["token_resolve"].append(resolved_span["text"])

        return result

    def _get_chain_mentions(self, chain) -> list:
        """
        Return mentions from a Coreferee chain.

        Coreferee chains behave like lists, but some versions also expose
        a .mentions attribute. This helper supports both variants.
        """
        if hasattr(chain, "mentions"):
            return list(chain.mentions)

        return list(chain)

    def _mention_to_external_span(
            self,
            doc: Doc,
            mention,
            expand_noun_chunks: bool,
    ) -> dict[str, Any]:
        """
        Convert a Coreferee mention to external begin/end offsets.

        A Coreferee mention is usually a list of token indices, for example:
            [14]
            [16, 19]
        """
        token_indices = self._mention_to_token_indices(mention)

        if not token_indices:
            raise ValueError("Coreferee mention does not contain token indices.")

        if expand_noun_chunks:
            token_indices = self._expand_indices_to_noun_chunks(
                doc=doc,
                token_indices=token_indices,
            )

        first_i = min(token_indices)
        last_i = max(token_indices)

        offsets = doc._.external_token_offsets

        begin = offsets[first_i]["begin"]
        end = offsets[last_i]["end"]

        contiguous_indices = list(range(first_i, last_i + 1))
        is_contiguous = token_indices == contiguous_indices

        if is_contiguous:
            text = doc[first_i:last_i + 1].text
        else:
            text = " ".join(doc[i].text for i in token_indices)

        return {
            "begin": begin,
            "end": end,
            "text": text,
            "token_indices": token_indices,
        }

    def _mention_to_token_indices(self, mention) -> list[int]:
        """
        Normalize a Coreferee mention to a list of token indices.
        """
        if mention is None:
            return []

        if hasattr(mention, "token_indexes"):
            return [int(i) for i in mention.token_indexes]

        if hasattr(mention, "token_indices"):
            return [int(i) for i in mention.token_indices]

        if isinstance(mention, int):
            return [int(mention)]

        return [int(i) for i in mention]

    def _expand_indices_to_noun_chunks(
            self,
            doc: Doc,
            token_indices: list[int],
    ) -> list[int]:
        """
        Expand token indices to their noun chunks if possible.

        Examples:
            token index for "cactus" -> indices for "a cactus"
            token index for "vase"   -> indices for "The vase"
        """
        expanded_indices = set(token_indices)

        try:
            noun_chunks = list(doc.noun_chunks)
        except Exception:
            return sorted(expanded_indices)

        for token_index in token_indices:
            for chunk in noun_chunks:
                if chunk.start <= token_index < chunk.end:
                    expanded_indices.update(range(chunk.start, chunk.end))
                    break

        return sorted(expanded_indices)

    @staticmethod
    def _infer_spaces_from_offsets(
            begins: list[int],
            ends: list[int],
    ) -> list[bool]:
        """
        Infer spaCy spaces from begin/end offsets.

        If the next token starts after the current token ends,
        there is whitespace between them.
        """
        spaces: list[bool] = []

        for i in range(len(begins)):
            if i == len(begins) - 1:
                spaces.append(False)
            else:
                spaces.append(ends[i] < begins[i + 1])

        return spaces

    @staticmethod
    def _validate_token_offsets(
            tokens: list[str],
            begins: list[int],
            ends: list[int],
    ) -> None:
        """
        Validate that tokens, begins and ends are aligned.
        """
        if not tokens:
            raise ValueError("tokens must not be empty.")

        if len(tokens) != len(begins) or len(tokens) != len(ends):
            raise ValueError(
                "tokens, begins and ends must have the same length. "
                f"tokens={len(tokens)}, begins={len(begins)}, ends={len(ends)}"
            )

        for i, (token, begin, end) in enumerate(zip(tokens, begins, ends)):
            if not token:
                raise ValueError(f"token must not be empty at index {i}.")

            if begin < 0:
                raise ValueError(f"begin must be >= 0 at index {i}.")

            if end < begin:
                raise ValueError(f"end must be >= begin at index {i}.")

            if i > 0 and begin < ends[i - 1]:
                raise ValueError(
                    f"Token offsets must not overlap. "
                    f"Problem at index {i}: begin={begin}, previous_end={ends[i - 1]}"
                )


if __name__ == "__main__":
    resolver = CorefereeResolver("en", "sm")

    tokens = [
        "Anna", "bought", "a", "cactus", ".",
        "The", "plant", "needed", "sunlight", ".",
        "She", "put", "a", "vase", "on", "the", "table", ".",
        "The", "vase", "was", "old", ",", "but", "it", "was", "beautiful", ".",
        "The", "cactus", "grew", "quickly", "because", "it", "got", "enough", "light", ".",
    ]

    begins = [
        0, 5, 12, 14, 20,
        22, 26, 32, 39, 47,
        49, 53, 57, 59, 64, 67, 71, 76,
        78, 82, 87, 91, 94, 96, 100, 103, 107, 116,
        118, 122, 129, 134, 142, 150, 153, 157, 164, 169,
    ]

    ends = [
        4, 11, 13, 20, 21,
        25, 31, 38, 47, 48,
        52, 56, 58, 63, 66, 70, 76, 77,
        81, 86, 90, 94, 95, 99, 102, 106, 116, 117,
        121, 128, 133, 141, 149, 152, 156, 163, 169, 170,
    ]

    doc = resolver.process_tokens(
        tokens=tokens,
        begins=begins,
        ends=ends,
    )

    print("Coreference dictionary:")
    result = resolver.get_coreference_dict(
        doc,
        include_self=False,
        expand_noun_chunks=True,
    )

    print(result)