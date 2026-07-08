from __future__ import annotations

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

from care import CareLocatorAgent, ParsedCareQuery


def _query(**overrides) -> ParsedCareQuery:
    base = dict(
        detected_language="English",
        response_language="English",
        summary="",
        medical_need=True,
        location="10013",
        specialties=[],
        insurance=[],
        preferred_languages=[],
        keywords=[],
        patient_context=None,
    )
    base.update(overrides)
    return ParsedCareQuery(**base)


class CareSettingEmergencyTests(unittest.TestCase):
    def setUp(self):
        self.agent = CareLocatorAgent(provider_search_service=Mock())

    def test_llm_care_setting_emergency_drives_emergency_even_without_english_keyword(self):
        # Benign English text (no "chest pain"/"911"); emergency comes from the LLM field.
        query = _query(care_setting="emergency", summary="needs help near 10013")
        self.assertEqual(self.agent._classify_care_setting(query, "needs help near 10013"), "emergency")

    def test_llm_urgency_emergency_drives_emergency(self):
        query = _query(urgency="emergency", summary="help near 10013")
        self.assertEqual(self.agent._classify_care_setting(query, "help near 10013"), "emergency")

    def test_english_keyword_still_triggers_emergency_without_llm_field(self):
        query = _query(care_setting=None, urgency=None)
        self.assertEqual(self.agent._classify_care_setting(query, "i have chest pain"), "emergency")

    def test_non_emergency_query_is_not_emergency(self):
        query = _query(care_setting="pcp", urgency="routine", specialties=["Primary Care"])
        self.assertNotEqual(self.agent._classify_care_setting(query, "routine checkup near 10013"), "emergency")


if __name__ == "__main__":
    unittest.main()
