import re
import unittest

from care.generate_locales import LOCALE_LANGUAGE_NAMES, build_english_master
from care.locales_loader import load_locales

_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


class LocaleFileInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = build_english_master()
        cls.locales = load_locales()

    def test_every_known_language_has_a_file(self):
        self.assertEqual(set(self.locales.keys()), set(LOCALE_LANGUAGE_NAMES.keys()))

    def test_copy_keys_match_master_exactly(self):
        for key, locale in self.locales.items():
            self.assertEqual(set(locale["copy"].keys()), set(self.master["copy"].keys()), key)
            self.assertEqual(
                set(locale["copy"]["phi_type_labels"].keys()),
                set(self.master["copy"]["phi_type_labels"].keys()),
                key,
            )

    def test_sentence_keys_match_master_exactly(self):
        for key, locale in self.locales.items():
            self.assertEqual(set(locale["sentences"].keys()), set(self.master["sentences"]), key)

    def test_placeholders_match_master_per_string(self):
        # Multiset (sorted) comparison: duplicate placeholders must survive.
        for key, locale in self.locales.items():
            for copy_key, master_value in self.master["copy"].items():
                if isinstance(master_value, dict):
                    continue
                self.assertEqual(
                    sorted(_PLACEHOLDER_RE.findall(master_value)),
                    sorted(_PLACEHOLDER_RE.findall(locale["copy"][copy_key])),
                    (key, copy_key),
                )

    def test_trust_guidance_and_mark_present(self):
        for key, locale in self.locales.items():
            self.assertTrue(locale["trust_guidance"].strip(), key)
            self.assertTrue(locale["auto_translated_mark"].strip(), key)

    def test_trust_guidance_keeps_master_line_structure(self):
        # A silently truncated footer (max_tokens) would drop bullet lines.
        master_lines = self.master["trust_guidance"].count("\n")
        for key, locale in self.locales.items():
            self.assertEqual(locale["trust_guidance"].count("\n"), master_lines, key)


if __name__ == "__main__":
    unittest.main()
