from __future__ import annotations

import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/main/python"))

from duui_opf_core import (
    DEFAULT_PLACEHOLDER,
    DEFAULT_MODE,
    PSEUDO_MODE,
    RedactionSpan,
    SelectionRange,
    apply_replacement_text,
    apply_selection,
    compose_selection_output,
    resolve_selection,
    split_options,
)


class DuuiOpfCoreTests(unittest.TestCase):
    def test_split_options_separates_service_and_decode_values(self) -> None:
        service_options, decode_options, mode, placeholder = split_options(
            {
                "model": "local-checkpoint",
                "context_window_length": 128,
                "trim_whitespace": False,
                "device": "cpu",
                "output_mode": "typed",
                "discard_overlapping_predicted_spans": True,
                "mode": PSEUDO_MODE,
                "placeholder": "<MASK>",
                "decode_mode": "argmax",
                "calibration_path": "/tmp/calibration.json",
                "selection_begin": 2,
                "selection_end": 8,
            }
        )

        self.assertEqual(service_options["model"], "local-checkpoint")
        self.assertEqual(service_options["device"], "cpu")
        self.assertEqual(decode_options["decode_mode"], "argmax")
        self.assertEqual(decode_options["viterbi_calibration_path"], "/tmp/calibration.json")
        self.assertEqual(mode, PSEUDO_MODE)
        self.assertEqual(placeholder, "<MASK>")

    def test_resolve_selection_accepts_nested_or_flat_offsets(self) -> None:
        nested = resolve_selection({"selection": {"begin": 4, "end": 9}}, text_length=20)
        flat = resolve_selection({"selection_begin": 1, "selection_end": 3}, text_length=20)

        self.assertEqual(nested, SelectionRange(begin=4, end=9))
        self.assertEqual(flat, SelectionRange(begin=1, end=3))

    def test_apply_replacement_text_uses_one_placeholder(self) -> None:
        redacted = apply_replacement_text(
            "Alice called Bob.",
            [
                RedactionSpan(label="private_person", start=0, end=5, text="Alice"),
                RedactionSpan(label="private_person", start=13, end=16, text="Bob"),
            ],
        )

        self.assertEqual(redacted, f"{DEFAULT_PLACEHOLDER} called {DEFAULT_PLACEHOLDER}.")

    def test_apply_selection_and_compose_output(self) -> None:
        selection = SelectionRange(begin=6, end=11)
        selected_text, offset = apply_selection("hello world", selection)

        self.assertEqual(selected_text, "world")
        self.assertEqual(offset, 6)
        self.assertEqual(compose_selection_output("hello world", selection, "there"), "hello there")


if __name__ == "__main__":
    unittest.main()
