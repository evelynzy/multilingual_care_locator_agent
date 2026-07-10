"""Gate wiring tests: handle_request redacts message + history before any LLM call."""

import sys
import types
import unittest
from unittest.mock import patch

if "huggingface_hub" not in sys.modules:
    huggingface_stub = types.ModuleType("huggingface_hub")

    class _StubInferenceClient:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("InferenceClient stub should not be used in tests")

    huggingface_stub.InferenceClient = _StubInferenceClient
    sys.modules["huggingface_hub"] = huggingface_stub

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    def _stub_get(*args, **kwargs):
        raise RuntimeError("requests.get stub should be patched in tests")

    requests_stub.get = _stub_get
    sys.modules["requests"] = requests_stub

from care import CareLocatorAgent, ParsedCareQuery
from care.rendering import _phi_notice_line


def _parsed(**overrides):
    values = dict(
        detected_language="English",
        response_language="English",
        summary="care request",
        medical_need=True,
        location="94110",
        specialties=["cardiology"],
        insurance=[],
        preferred_languages=[],
        keywords=[],
        patient_context=None,
        urgency=None,
        care_setting=None,
        needs_clarification=False,
        follow_up_focus=[],
    )
    values.update(overrides)
    return ParsedCareQuery(**values)


_CLARIFY_GUIDANCE = {
    "mode": "search",
    "care_setting_guidance": None,
    "follow_up_questions": ["Which area?"],
    "specialist_plan_guidance": None,
    "location_only": True,
}


class GateWiringTests(unittest.TestCase):
    def setUp(self):
        self.agent = CareLocatorAgent()

    def test_interpret_sees_redacted_message_and_history(self):
        seen = {}

        def capture(client, message, history):
            seen.setdefault("calls", []).append(
                (message, [t.get("content") for t in history])
            )
            return _parsed()

        with patch.object(CareLocatorAgent, "_interpret_user_need", side_effect=capture), \
             patch.object(CareLocatorAgent, "_compose_response", return_value="REPLY"), \
             patch.object(CareLocatorAgent, "_build_navigation_guidance", return_value=dict(_CLARIFY_GUIDANCE)):
            history = [
                {"role": "user", "content": "my ssn is 123-45-6789"},
                {"role": "assistant", "content": "ok"},
            ]
            self.agent.handle_request(
                client=object(), message="phone 415-555-0123, cardiology 94110",
                history=history, max_tokens=64, temperature=0.1, top_p=0.9,
            )

        message_arg, history_contents = seen["calls"][0]
        self.assertIn("[REDACTED: PHONE]", message_arg)
        self.assertNotIn("415-555-0123", message_arg)
        self.assertIn("[REDACTED: SSN]", history_contents[0])
        self.assertNotIn("123-45-6789", history_contents[0])
        self.assertEqual(history_contents[1], "ok")  # assistant turns untouched

    def test_notice_prepended_when_current_message_has_phi(self):
        with patch.object(CareLocatorAgent, "_interpret_user_need", return_value=_parsed()), \
             patch.object(CareLocatorAgent, "_compose_response", return_value="REPLY"), \
             patch.object(CareLocatorAgent, "_build_navigation_guidance", return_value=dict(_CLARIFY_GUIDANCE)):
            reply = self.agent.handle_request(
                client=object(), message="ssn 123-45-6789 cardiology 94110",
                history=[], max_tokens=64, temperature=0.1, top_p=0.9,
            )
        self.assertTrue(reply.startswith("🔒"))
        self.assertIn("REPLY", reply)

    def test_no_notice_for_history_only_phi(self):
        with patch.object(CareLocatorAgent, "_interpret_user_need", return_value=_parsed()), \
             patch.object(CareLocatorAgent, "_compose_response", return_value="REPLY"), \
             patch.object(CareLocatorAgent, "_build_navigation_guidance", return_value=dict(_CLARIFY_GUIDANCE)):
            reply = self.agent.handle_request(
                client=object(), message="94110",
                history=[{"role": "user", "content": "my ssn is 123-45-6789"}],
                max_tokens=64, temperature=0.1, top_p=0.9,
            )
        self.assertFalse(reply.startswith("🔒"))

    def test_clean_message_reply_unchanged(self):
        with patch.object(CareLocatorAgent, "_interpret_user_need", return_value=_parsed()), \
             patch.object(CareLocatorAgent, "_compose_response", return_value="REPLY"), \
             patch.object(CareLocatorAgent, "_build_navigation_guidance", return_value=dict(_CLARIFY_GUIDANCE)):
            reply = self.agent.handle_request(
                client=object(), message="primary care 10001",
                history=[], max_tokens=64, temperature=0.1, top_p=0.9,
            )
        self.assertEqual(reply, "REPLY")


class PhiNoticeLineTests(unittest.TestCase):
    def test_english_notice_names_types(self):
        line = _phi_notice_line(["ssn", "phone"], "english")
        self.assertTrue(line.startswith("🔒"))
        self.assertIn("Social Security number", line)
        self.assertIn("phone number", line)

    def test_known_languages_have_native_copy(self):
        from care.rendering import _DETERMINISTIC_RENDER_COPY

        for key in ("spanish", "simplified_chinese", "korean", "arabic"):
            expected_label = _DETERMINISTIC_RENDER_COPY[key]["phi_type_labels"]["ssn"]
            self.assertIn(expected_label, _phi_notice_line(["ssn"], key))

    def test_unknown_language_key_falls_back_to_english(self):
        self.assertIn("Social Security number", _phi_notice_line(["ssn"], "russian"))


if __name__ == "__main__":
    unittest.main()
