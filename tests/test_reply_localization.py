"""Restore any-language localization of the provider-results reply (regression from
df2362c, which made cards deterministic but en/es/zh-only). The wrapper copy is
LLM-translated into the user's language while provider data stays verbatim."""

from __future__ import annotations

import unittest

from care import CareLocatorAgent, _reply_localization_target


class ReplyLocalizationTargetTests(unittest.TestCase):
    def test_english_needs_no_localization(self):
        self.assertIsNone(_reply_localization_target("English"))
        self.assertIsNone(_reply_localization_target("en"))

    def test_natively_supported_languages_need_no_llm(self):
        self.assertIsNone(_reply_localization_target("Spanish"))
        self.assertIsNone(_reply_localization_target("Chinese"))

    def test_unknown_or_empty_needs_no_localization(self):
        self.assertIsNone(_reply_localization_target(""))
        self.assertIsNone(_reply_localization_target(None))

    def test_arbitrary_language_is_localized(self):
        self.assertEqual(_reply_localization_target("Czech"), "Czech")

    def test_language_with_footer_only_support_is_localized(self):
        # Korean/Arabic have a safety-footer translation but no card-body copy,
        # so the wrapper still falls back to English -> localize.
        self.assertEqual(_reply_localization_target("Korean"), "Korean")
        self.assertEqual(_reply_localization_target("Arabic"), "Arabic")


class _StubClient:
    def __init__(self, content, raise_exc=False):
        self.content = content
        self.raise_exc = raise_exc
        self.calls = []

    def chat_completion(self, messages, max_tokens=None, temperature=None, top_p=None, **kwargs):
        self.calls.append(messages)
        if self.raise_exc:
            raise RuntimeError("boom")

        class _Msg:
            content = self.content

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class LocalizeViaLlmTests(unittest.TestCase):
    def _agent(self):
        return CareLocatorAgent()

    def test_returns_translation_from_client(self):
        client = _StubClient("Zde jsou výsledky. 1. ADAM DETORA")
        out = self._agent()._localize_reply_via_llm(
            client, "Here are results. 1. ADAM DETORA", "Czech", 800, 0.0, 1.0
        )
        self.assertEqual(out, "Zde jsou výsledky. 1. ADAM DETORA")

    def test_prompt_preserves_data_and_names_target(self):
        client = _StubClient("x")
        self._agent()._localize_reply_via_llm(
            client, "1. ADAM DETORA — Pediatrics", "Czech", 800, 0.0, 1.0
        )
        prompt = str(client.calls[0])
        self.assertIn("Czech", prompt)
        self.assertIn("EXACTLY", prompt)
        self.assertIn("ADAM DETORA", prompt)

    def test_empty_reply_skips_client(self):
        client = _StubClient("x")
        out = self._agent()._localize_reply_via_llm(client, "", "Czech", 800, 0.0, 1.0)
        self.assertEqual(out, "")
        self.assertEqual(client.calls, [])

    def test_client_error_falls_back_to_original(self):
        client = _StubClient("x", raise_exc=True)
        original = "Here are results. 1. ADAM DETORA"
        out = self._agent()._localize_reply_via_llm(client, original, "Czech", 800, 0.0, 1.0)
        self.assertEqual(out, original)

    def test_empty_completion_falls_back_to_original(self):
        client = _StubClient("")  # blank content
        original = "Here are results. 1. ADAM DETORA"
        out = self._agent()._localize_reply_via_llm(client, original, "Czech", 800, 0.0, 1.0)
        self.assertEqual(out, original)


if __name__ == "__main__":
    unittest.main()
