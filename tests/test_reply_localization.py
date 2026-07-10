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

    def test_locale_file_languages_need_no_llm(self):
        # Korean/Arabic (and vi/tl) now render natively from locale files,
        # so the LLM wrapper-translation pass is reserved for the long tail.
        self.assertIsNone(_reply_localization_target("Korean"))
        self.assertIsNone(_reply_localization_target("Arabic"))
        self.assertIsNone(_reply_localization_target("Vietnamese"))
        self.assertIsNone(_reply_localization_target("Tagalog"))


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


class _SequenceClient:
    """Each queued item is an Exception (raised) or a content string."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def chat_completion(self, messages, max_tokens=None, temperature=None, top_p=None, **kwargs):
        self.calls.append(messages)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item

        class _Msg:
            content = item

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class LocalizeRetryAndTelemetryTests(unittest.TestCase):
    def _agent(self):
        return CareLocatorAgent()

    def test_retries_once_after_error_then_succeeds(self):
        client = _SequenceClient([RuntimeError("boom"), "Zde jsou výsledky. 94110"])
        agent = self._agent()
        out = agent._localize_reply_via_llm(
            client, "Here are results. 94110", "Czech", 800, 0.0, 1.0
        )
        self.assertEqual(out, "Zde jsou výsledky. 94110")
        self.assertEqual(len(client.calls), 2)
        self.assertIsNone(getattr(agent, "last_localization_fallback", None))

    def test_english_echo_is_rejected_and_retried(self):
        original = "Here are results. 1. ADAM DETORA"
        client = _SequenceClient([original, "Zde jsou výsledky. 1. ADAM DETORA"])
        agent = self._agent()
        out = agent._localize_reply_via_llm(client, original, "Czech", 800, 0.0, 1.0)
        self.assertEqual(out, "Zde jsou výsledky. 1. ADAM DETORA")
        self.assertEqual(len(client.calls), 2)

    def test_double_failure_falls_back_and_flags(self):
        client = _SequenceClient([RuntimeError("boom"), RuntimeError("boom")])
        agent = self._agent()
        original = "Here are results. 1. ADAM DETORA"
        out = agent._localize_reply_via_llm(client, original, "Czech", 800, 0.0, 1.0)
        self.assertEqual(out, original)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(agent.last_localization_fallback, "Czech")

    def test_source_sent_to_client_carries_translated_mark(self):
        client = _StubClient("Zde jsou výsledky.")
        self._agent()._localize_reply_via_llm(
            client, "Here are results.", "Czech", 800, 0.0, 1.0
        )
        self.assertIn("auto-translated from English", str(client.calls[0]))


if __name__ == "__main__":
    unittest.main()
