"""F6: when a language-concordant provider is requested but none can be confirmed
to speak it, the reply must disclose the unmet need instead of silently listing
non-matching providers."""

from __future__ import annotations

import unittest

from care_agent import CareLocatorAgent


class _FakeProvider:
    def __init__(self, languages):
        self.languages = languages


class _FakeResult:
    def __init__(self, languages=()):
        self.provider = _FakeProvider(languages)


class UnverifiedLanguageDetectionTests(unittest.TestCase):
    def test_discloses_when_no_provider_speaks_requested_language(self):
        result = CareLocatorAgent._unverified_preferred_languages(
            ["Spanish"], [_FakeResult(()), _FakeResult(())]
        )
        self.assertEqual(result, ["Spanish"])

    def test_no_disclosure_when_a_provider_speaks_it(self):
        result = CareLocatorAgent._unverified_preferred_languages(
            ["Spanish"], [_FakeResult(("Spanish",)), _FakeResult(())]
        )
        self.assertEqual(result, [])

    def test_match_is_case_insensitive(self):
        result = CareLocatorAgent._unverified_preferred_languages(
            ["Spanish"], [_FakeResult(("spanish",))]
        )
        self.assertEqual(result, [])

    def test_nothing_requested_returns_empty(self):
        self.assertEqual(
            CareLocatorAgent._unverified_preferred_languages([], [_FakeResult(())]), []
        )

    def test_only_unmatched_languages_are_returned(self):
        result = CareLocatorAgent._unverified_preferred_languages(
            ["Spanish", "French"], [_FakeResult(("Spanish",))]
        )
        self.assertEqual(result, ["French"])


class DisclosureRenderTests(unittest.TestCase):
    def _payload(self, language_unverified=None, response_language="English"):
        return {
            "query": {
                "summary": "Spanish-speaking primary care 90011",
                "response_language": response_language,
            },
            "local_results": [],
            "fallback_results": [],
            "language_unverified": language_unverified,
        }

    def test_note_present_when_flagged(self):
        agent = CareLocatorAgent()
        out = agent._compose_result_card_response(self._payload(["Spanish"]))
        self.assertIn("Spanish", out)
        self.assertIn("could not confirm", out.lower())

    def test_note_absent_when_not_flagged(self):
        agent = CareLocatorAgent()
        out = agent._compose_result_card_response(self._payload(None))
        self.assertNotIn("could not confirm", out.lower())

    def test_note_localized_in_spanish(self):
        agent = CareLocatorAgent()
        out = agent._compose_result_card_response(
            self._payload(["Spanish"], response_language="Spanish")
        )
        self.assertIn("No pudimos confirmar", out)


if __name__ == "__main__":
    unittest.main()
