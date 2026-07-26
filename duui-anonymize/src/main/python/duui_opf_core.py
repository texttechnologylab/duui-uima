from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_MODE = "replacement"
PSEUDO_MODE = "pseudo"
DEFAULT_PLACEHOLDER = "<REDACTED>"

_SELECTION_KEYS = {
    "selection",
    "selection_begin",
    "selection_end",
    "selection_start",
    "selection_stop",
}

_SERVICE_OPTION_KEYS = {
    "model",
    "context_window_length",
    "trim_whitespace",
    "device",
    "output_mode",
    "discard_overlapping_predicted_spans",
    "mode",
    "placeholder",
}

_DECODE_OPTION_KEYS = {
    "decode_mode",
    "viterbi_calibration_path",
    "calibration_path",
}


@dataclass(frozen=True)
class SelectionRange:
    begin: int
    end: int


@dataclass(frozen=True)
class RedactionSpan:
    label: str
    start: int
    end: int
    text: str
    placeholder: str = DEFAULT_PLACEHOLDER


def split_options(
    options: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    service_options: dict[str, Any] = {}
    decode_options: dict[str, Any] = {}
    mode = DEFAULT_MODE
    placeholder = DEFAULT_PLACEHOLDER

    for key, value in options.items():
        if key in _SELECTION_KEYS:
            continue
        if key == "mode":
            mode = str(value)
        elif key == "placeholder":
            placeholder = str(value)
        elif key in _SERVICE_OPTION_KEYS:
            service_options[key] = value
        elif key == "decode":
            continue
        elif key in _DECODE_OPTION_KEYS:
            if key == "calibration_path":
                decode_options["viterbi_calibration_path"] = value
            else:
                decode_options[key] = value

    return service_options, decode_options, mode, placeholder


def resolve_selection(
    options: Mapping[str, Any],
    *,
    text_length: int,
) -> SelectionRange | None:
    selection = options.get("selection")
    if isinstance(selection, Mapping):
        begin = selection.get("begin", selection.get("start"))
        end = selection.get("end", selection.get("stop"))
        if begin is None or end is None:
            return None
        return _validate_selection(begin, end, text_length=text_length)

    begin = options.get("selection_begin", options.get("selection_start"))
    end = options.get("selection_end", options.get("selection_stop"))
    if begin is None or end is None:
        return None
    return _validate_selection(begin, end, text_length=text_length)


def _validate_selection(
    begin: Any,
    end: Any,
    *,
    text_length: int,
) -> SelectionRange:
    begin_int = int(begin)
    end_int = int(end)
    if begin_int < 0 or end_int < begin_int or end_int > text_length:
        raise ValueError("selection must satisfy 0 <= begin <= end <= text length")
    return SelectionRange(begin=begin_int, end=end_int)


def apply_replacement_text(
    text: str,
    spans: list[RedactionSpan],
    *,
    placeholder: str = DEFAULT_PLACEHOLDER,
) -> str:
    if not spans:
        return text

    redacted_parts: list[str] = []
    cursor = 0
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if span.start < cursor:
            continue
        redacted_parts.append(text[cursor:span.start])
        redacted_parts.append(placeholder)
        cursor = max(cursor, span.end)
    redacted_parts.append(text[cursor:])
    return "".join(redacted_parts)


def apply_selection(
    text: str,
    selection: SelectionRange | None,
) -> tuple[str, int]:
    if selection is None:
        return text, 0
    return text[selection.begin:selection.end], selection.begin


def compose_selection_output(
    text: str,
    selection: SelectionRange | None,
    replacement: str,
) -> str:
    if selection is None:
        return replacement
    return text[:selection.begin] + replacement + text[selection.end:]
