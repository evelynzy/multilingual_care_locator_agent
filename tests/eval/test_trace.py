from __future__ import annotations

import os
import tempfile
import unittest

from eval.trace import Trace, TurnCapture, run_trace, trace_from_dict, trace_to_dict, _cache_path


class TraceSerializationTests(unittest.TestCase):
    def _sample(self):
        turn = TurnCapture(
            user_message="cardiology 95051",
            parsed_specialties=["cardiology"],
            parsed_preferred_languages=[],
            parsed_urgency=None,
            parsed_care_setting=None,
            parsed_needs_clarification=False,
            searched=True,
            request_specialties=["cardiology"],
            request_preferred_languages=[],
            provider_states=["CA"],
            provider_count=1,
            html_has_card=True,
        )
        return Trace(scenario_id="s01-cardiology", language="en", turns=[turn], error=None)

    def test_roundtrip_dict(self):
        trace = self._sample()
        restored = trace_from_dict(trace_to_dict(trace))
        self.assertEqual(restored, trace)

    def test_cache_path_is_stable_and_language_specific(self):
        with tempfile.TemporaryDirectory() as d:
            p_en = _cache_path(d, "s01-cardiology", "en", ("cardiology 95051",))
            p_en2 = _cache_path(d, "s01-cardiology", "en", ("cardiology 95051",))
            p_zh = _cache_path(d, "s01-cardiology", "zh", ("cardiology 95051",))
            self.assertEqual(p_en, p_en2)
            self.assertNotEqual(p_en, p_zh)


class TraceLiveTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("RUN_EVAL") == "1" and bool(os.getenv("HF_TOKEN")),
        "set RUN_EVAL=1 and HF_TOKEN to run the live trace test",
    )
    def test_english_cardiology_trace_captures_artifacts(self):
        from huggingface_hub import InferenceClient
        from config_loader import get_chat_model_settings
        from provider_search.service import ProviderSearchService
        from provider_search.sources.clinicaltables import ClinicalTablesSource
        from eval.dataset import load_scenarios
        from eval.instrumented_agent import TracingAgent

        settings = get_chat_model_settings()
        client = InferenceClient(model=settings["model_id"], token=os.getenv("HF_TOKEN"))
        agent = TracingAgent(ProviderSearchService(clinicaltables_source=ClinicalTablesSource()))

        scenario = next(s for s in load_scenarios() if s.id == "s01-cardiology")
        trace = run_trace(scenario, "en", agent, client, settings, use_cache=False)

        self.assertIsNone(trace.error)
        self.assertEqual(len(trace.turns), 1)
        turn = trace.turns[0]
        self.assertTrue(turn.searched)
        self.assertIn("cardiology", " ".join(turn.request_specialties).lower())
        self.assertTrue(turn.html_has_card)


if __name__ == "__main__":
    unittest.main()
