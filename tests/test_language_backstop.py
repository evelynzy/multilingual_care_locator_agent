import unittest
from unittest.mock import Mock

# tests/ has no __init__.py: pytest puts the tests dir itself on sys.path,
# so sibling test modules import WITHOUT the tests. prefix.
from test_followup_state import _lang_query

from care import CareLocatorAgent
from care.language import _dominant_user_script_language, _message_has_language_signal


class MessageLanguageSignalTests(unittest.TestCase):
    def test_zip_codes_and_punctuation_carry_no_signal(self):
        for text in ("94110", "94110-1234", "  10013 ", "", "  ", "?!.", "123 456"):
            self.assertFalse(_message_has_language_signal(text), text)

    def test_any_letters_carry_signal(self):
        for text in ("ok", "si", "thanks, English please", "小儿医生", "안녕하세요", "مرحبا"):
            self.assertTrue(_message_has_language_signal(text), text)


class DominantScriptTests(unittest.TestCase):
    def test_han_dominant_returns_chinese(self):
        self.assertEqual(_dominant_user_script_language(["帮我找一个附近的小儿医生", "94110"]), "Chinese")

    def test_hangul_dominant_returns_korean(self):
        self.assertEqual(_dominant_user_script_language(["근처 소아과 의사를 찾아주세요"]), "Korean")

    def test_arabic_dominant_returns_arabic(self):
        self.assertEqual(_dominant_user_script_language(["أحتاج طبيب أطفال قريب"]), "Arabic")

    def test_latin_returns_none(self):
        self.assertIsNone(_dominant_user_script_language(["find me a doctor", "94110"]))

    def test_mixed_below_majority_returns_none(self):
        # Half Latin letters, half Han: no strict majority for any script bucket.
        self.assertIsNone(_dominant_user_script_language(["abcd", "医生看病"]))

    def test_empty_returns_none(self):
        self.assertIsNone(_dominant_user_script_language([]))
        self.assertIsNone(_dominant_user_script_language(["94110"]))


class ConversationLanguageBackstopTests(unittest.TestCase):
    def setUp(self):
        self.agent = CareLocatorAgent(provider_search_service=Mock())

    def test_overrides_english_merge_on_signal_less_turn_with_han_history(self):
        query = _lang_query("English", "English")
        history = [
            {"role": "user", "content": "帮我找一个附近的小儿医生"},
            {"role": "assistant", "content": "Please share your city or ZIP."},
        ]
        out = self.agent._apply_conversation_language_backstop(query, "94110", history)
        self.assertEqual(out.detected_language, "Chinese")
        self.assertEqual(out.response_language, "Chinese")

    def test_no_override_when_latest_turn_has_letters(self):
        query = _lang_query("English", "English")
        history = [{"role": "user", "content": "帮我找一个附近的小儿医生"}]
        out = self.agent._apply_conversation_language_backstop(
            query, "thanks, English please", history
        )
        self.assertEqual(out.response_language, "English")

    def test_no_override_when_merge_already_non_english(self):
        query = _lang_query("Chinese", "Chinese")
        history = [{"role": "user", "content": "帮我找一个附近的小儿医生"}]
        out = self.agent._apply_conversation_language_backstop(query, "94110", history)
        self.assertIs(out, query)

    def test_no_override_without_history(self):
        query = _lang_query("English", "English")
        out = self.agent._apply_conversation_language_backstop(query, "94110", [])
        self.assertIs(out, query)

    def test_no_override_for_latin_history(self):
        query = _lang_query("English", "English")
        history = [{"role": "user", "content": "find me a children's doctor"}]
        out = self.agent._apply_conversation_language_backstop(query, "94110", history)
        self.assertIs(out, query)

    def test_overrides_unknown_merge_with_hangul_history(self):
        query = _lang_query("unknown", "unknown")
        history = [{"role": "user", "content": "근처 소아과 의사를 찾아주세요"}]
        out = self.agent._apply_conversation_language_backstop(query, "94110", history)
        self.assertEqual(out.response_language, "Korean")


if __name__ == "__main__":
    unittest.main()
