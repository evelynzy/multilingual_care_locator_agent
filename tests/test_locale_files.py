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


class LocaleWiringTests(unittest.TestCase):
    def test_all_seven_languages_resolve_natively(self):
        from care.rendering import _reply_localization_target, _resolved_supported_language_key

        for language, key in (
            ("Spanish", "spanish"),
            ("Chinese", "simplified_chinese"),
            ("Arabic", "arabic"),
            ("Korean", "korean"),
            ("Vietnamese", "vietnamese"),
            ("Tagalog", "tagalog"),
        ):
            self.assertEqual(_resolved_supported_language_key(language), key)
            self.assertIsNone(_reply_localization_target(language))

    def test_long_tail_still_targets_llm_pass(self):
        from care.rendering import _reply_localization_target

        self.assertEqual(_reply_localization_target("Czech"), "Czech")

    def test_render_copy_serves_locale_strings(self):
        from care.rendering import _DETERMINISTIC_RENDER_COPY

        for key in ("arabic", "korean", "vietnamese", "tagalog"):
            self.assertIn(key, _DETERMINISTIC_RENDER_COPY)
            self.assertNotEqual(
                _DETERMINISTIC_RENDER_COPY[key]["results_intro"],
                _DETERMINISTIC_RENDER_COPY["english"]["results_intro"],
            )

    def test_sentence_translations_served_from_locales(self):
        from care.rendering import _DETERMINISTIC_RENDER_TRANSLATIONS

        sentence = "For routine or ongoing care, primary care is usually the best fit."
        for key in ("spanish", "simplified_chinese", "korean"):
            self.assertIn(key, _DETERMINISTIC_RENDER_TRANSLATIONS[sentence])

    def test_non_english_footers_carry_localized_mark(self):
        from care.safety import _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE

        locales = load_locales()
        for key, locale in locales.items():
            footer = _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE[key]
            self.assertIn(locale["auto_translated_mark"].strip(), footer, key)
        self.assertNotIn(
            "auto-translated", _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE["english"].lower()
        )

    def test_footer_lookup_still_alias_driven(self):
        from care.safety import (
            _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE,
            _get_prewritten_required_trust_guidance,
        )

        self.assertEqual(
            _get_prewritten_required_trust_guidance("한국어"),
            _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE["korean"],
        )


if __name__ == "__main__":
    unittest.main()
