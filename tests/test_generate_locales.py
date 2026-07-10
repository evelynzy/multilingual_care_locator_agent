import re
import unittest
from unittest.mock import Mock

from care.generate_locales import (
    LOCALE_LANGUAGE_NAMES,
    _translate_string,
    build_english_master,
    generate_locale,
)


def _mock_client(reply_text):
    completion = Mock()
    completion.choices = [Mock(message=Mock(content=reply_text))]
    client = Mock()
    client.chat_completion.return_value = completion
    return client


class EnglishMasterTests(unittest.TestCase):
    def test_master_has_all_sections(self):
        master = build_english_master()
        self.assertIn("results_intro", master["copy"])
        self.assertIn("phi_type_labels", master["copy"])
        self.assertTrue(any("urgent care" in s for s in master["sentences"]))
        self.assertIn("safety and trust", master["trust_guidance"].lower())
        self.assertIn("auto-translated from English", master["auto_translated_mark"])

    def test_known_language_set(self):
        self.assertEqual(
            set(LOCALE_LANGUAGE_NAMES),
            {"spanish", "simplified_chinese", "arabic", "korean", "vietnamese", "tagalog"},
        )


class TranslateStringTests(unittest.TestCase):
    def test_placeholders_survive(self):
        client = _mock_client("X {summary} Y")
        out = _translate_string("Results for {summary}.", "Spanish", client, "m")
        self.assertIn("{summary}", out)

    def test_mangled_placeholder_raises(self):
        client = _mock_client("X summary Y")
        with self.assertRaises(ValueError):
            _translate_string("Results for {summary}.", "Spanish", client, "m")

    def test_empty_translation_raises(self):
        client = _mock_client("")
        with self.assertRaises(ValueError):
            _translate_string("Results.", "Spanish", client, "m")

    def test_dropped_duplicate_placeholder_raises(self):
        client = _mock_client("X {value} Y")
        with self.assertRaises(ValueError):
            _translate_string("{value} and {value}", "Spanish", client, "m")

    def test_lost_line_structure_raises(self):
        client = _mock_client("one line only")
        with self.assertRaises(ValueError):
            _translate_string("line one\nline two", "Spanish", client, "m")

    def test_lost_label_text_raises(self):
        client = _mock_client("{value}")
        with self.assertRaises(ValueError):
            _translate_string("Source: {value}", "Spanish", client, "m")


class GenerateLocaleTests(unittest.TestCase):
    def test_locale_mirrors_master_keys(self):
        def _echo_placeholders(messages, **kwargs):
            # Return a fake translation that carries EXACTLY the source's
            # placeholders AND newline count, so every guard passes for every
            # template (incl. the multi-line trust footer).
            source = messages[-1]["content"]
            tokens = " ".join(re.findall(r"\{[a-z_]+\}", source))
            # "\nx" (not bare "\n") so the newlines survive _extract_text's strip()
            content = ("T " + tokens).strip() + "\nx" * source.count("\n")
            return Mock(choices=[Mock(message=Mock(content=content))])

        client = Mock()
        client.chat_completion.side_effect = _echo_placeholders
        locale = generate_locale("korean", client, "m")
        master = build_english_master()
        self.assertEqual(set(locale["copy"].keys()), set(master["copy"].keys()))
        self.assertEqual(
            set(locale["copy"]["phi_type_labels"].keys()),
            set(master["copy"]["phi_type_labels"].keys()),
        )
        self.assertEqual(set(locale["sentences"].keys()), set(master["sentences"]))
        self.assertTrue(locale["trust_guidance"])
        self.assertTrue(locale["auto_translated_mark"])
        self.assertEqual(locale["language_key"], "korean")


if __name__ == "__main__":
    unittest.main()
