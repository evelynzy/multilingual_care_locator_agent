from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import Mock

if "huggingface_hub" not in sys.modules:
    _hf = types.ModuleType("huggingface_hub")

    class _StubInferenceClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InferenceClient stub should not be used in tests")

    _hf.InferenceClient = _StubInferenceClient
    sys.modules["huggingface_hub"] = _hf

from care_agent import CareLocatorAgent, INTERPRET_MAX_TOKENS


def _completion(content: str):
    choice = type("Choice", (), {"message": {"content": content}, "finish_reason": "stop"})()
    return type("Completion", (), {"choices": [choice]})()


_VALID = json.dumps(
    {
        "detected_language": "Chinese",
        "response_language": "中文",
        "summary": "pediatrics 10013",
        "medical_need": True,
        "location": "10013",
        "specialties": ["pediatrics"],
        "insurance": [],
        "preferred_languages": [],
        "keywords": [],
        "patient_context": None,
        "care_setting": "specialist",
        "urgency": None,
        "needs_clarification": False,
        "follow_up_focus": [],
    }
)


class InterpretStructuredOutputTests(unittest.TestCase):
    def setUp(self):
        self.agent = CareLocatorAgent(provider_search_service=Mock())

    def test_interpret_requests_json_schema_structured_output(self):
        client = Mock()
        client.chat_completion.return_value = _completion(_VALID)

        self.agent._interpret_user_need(client, "儿科10013", [])

        _, kwargs = client.chat_completion.call_args
        self.assertEqual(kwargs["response_format"]["type"], "json_schema")
        self.assertTrue(kwargs["response_format"]["json_schema"]["strict"])

    def test_interpret_uses_higher_max_tokens(self):
        client = Mock()
        client.chat_completion.return_value = _completion(_VALID)

        self.agent._interpret_user_need(client, "儿科10013", [])

        _, kwargs = client.chat_completion.call_args
        self.assertGreaterEqual(kwargs["max_tokens"], 1024)
        self.assertEqual(kwargs["max_tokens"], INTERPRET_MAX_TOKENS)

    def test_interpret_parses_english_specialty_and_location(self):
        client = Mock()
        client.chat_completion.return_value = _completion(_VALID)

        result = self.agent._interpret_user_need(client, "儿科10013", [])

        self.assertEqual(result.specialties, ["pediatrics"])
        self.assertEqual(result.location, "10013")


class InterpretGracefulDegradationTests(unittest.TestCase):
    """Provider rejects response_format → must fall back to plain call, not raise."""

    def setUp(self):
        self.agent = CareLocatorAgent(provider_search_service=Mock())

    def _make_rejecting_client(self):
        """Returns a client whose chat_completion raises when response_format is present,
        but returns a valid completion otherwise."""

        def _chat_completion(messages, **kwargs):
            if "response_format" in kwargs:
                raise RuntimeError("response_format not supported")
            return _completion(_VALID)

        client = Mock()
        client.chat_completion.side_effect = _chat_completion
        return client

    def test_degrade_gracefully_when_provider_rejects_structured_output(self):
        """If response_format causes an exception, the call must be retried without it
        and _interpret_user_need must return a valid result rather than raising."""
        client = self._make_rejecting_client()

        result = self.agent._interpret_user_need(client, "儿科10013", [])

        self.assertEqual(result.location, "10013")

    def test_at_least_one_call_made_without_response_format(self):
        """After the structured-output call fails, at least one call without
        response_format must have been attempted."""
        client = self._make_rejecting_client()

        self.agent._interpret_user_need(client, "儿科10013", [])

        calls_without_rf = [
            call
            for call in client.chat_completion.call_args_list
            if "response_format" not in call[1]
        ]
        self.assertGreater(
            len(calls_without_rf),
            0,
            "Expected at least one call without response_format",
        )


if __name__ == "__main__":
    unittest.main()
