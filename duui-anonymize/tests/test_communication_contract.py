from __future__ import annotations

from pathlib import Path
import unittest


class CommunicationContractTests(unittest.TestCase):
    def test_lua_contract_mentions_text_options_selection_and_redacted_view(self) -> None:
        lua_path = Path(__file__).resolve().parents[1] / "src/main/python/communication.lua"
        contents = lua_path.read_text(encoding="utf-8")

        self.assertIn('text = text', contents)
        self.assertIn('options = copy_options(params)', contents)
        self.assertIn('selection = resolve_selection(params)', contents)
        self.assertIn('createView("opf_redacted")', contents)
        self.assertIn('detected_spans', contents)


if __name__ == "__main__":
    unittest.main()
