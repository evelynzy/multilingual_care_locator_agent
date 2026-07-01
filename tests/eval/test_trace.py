from __future__ import annotations

import os
import tempfile
import unittest

from eval.trace import Trace, TurnCapture, run_trace, trace_from_dict, trace_to_dict, _cache_path


class TraceSerializationTests(unittest.TestCase):
    def _sample(self):
        turn = TurnCapture(
            user_message="cardiology 98101",
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
            emergency_routed=False,
        )
        return Trace(scenario_id="s01-cardiology", language="en", turns=[turn], error=None)

    def test_roundtrip_dict(self):
        trace = self._sample()
        restored = trace_from_dict(trace_to_dict(trace))
        self.assertEqual(restored, trace)

    def test_cache_path_is_stable_and_language_specific(self):
        with tempfile.TemporaryDirectory() as d:
            p_en = _cache_path(d, "s01-cardiology", "en", ("cardiology 98101",), "model-a")
            p_en2 = _cache_path(d, "s01-cardiology", "en", ("cardiology 98101",), "model-a")
            p_zh = _cache_path(d, "s01-cardiology", "zh", ("cardiology 98101",), "model-a")
            self.assertEqual(p_en, p_en2)
            self.assertNotEqual(p_en, p_zh)
            p_model_a = _cache_path(d, "s01-cardiology", "en", ("cardiology 98101",), "model-a")
            p_model_b = _cache_path(d, "s01-cardiology", "en", ("cardiology 98101",), "model-b")
            self.assertNotEqual(p_model_a, p_model_b)


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


class _FakeProvider:
    def __init__(self, state=None, address=None):
        self.state = state
        self.address = address
        self.location_summary = address


class _FakeResult:
    def __init__(self, state):
        self.provider = _FakeProvider(state)


class _FakeResponse:
    def __init__(self, results):
        self.provider_results = results


class _FakeService:
    def __init__(self, last_request=None, last_response=None):
        self.last_request = last_request
        self.last_response = last_response

    def reset(self):
        self.last_request = None
        self.last_response = None


class _FakeParsed:
    def __init__(self, specialties=None, preferred_languages=None, urgency=None,
                 care_setting=None, needs_clarification=False):
        self.specialties = specialties or []
        self.preferred_languages = preferred_languages or []
        self.urgency = urgency
        self.care_setting = care_setting
        self.needs_clarification = needs_clarification


class _CaptureAgent:
    def __init__(self, parsed=None, service=None, last_navigation_mode=None):
        self.last_parsed_query = parsed
        self.provider_search_service = service or _FakeService()
        self.last_navigation_mode = last_navigation_mode


class _RaisingAgent:
    def __init__(self):
        self.last_parsed_query = None
        self.last_navigation_mode = None
        self.provider_search_service = _FakeService()

    def reset_capture(self):
        self.last_parsed_query = None
        self.provider_search_service.reset()

    def handle_request(self, *args, **kwargs):
        raise RuntimeError("boom")


class TraceCaptureAndCacheTests(unittest.TestCase):
    def test_capture_no_search_turn(self):
        from eval.trace import _capture_turn

        agent = _CaptureAgent(
            parsed=_FakeParsed(needs_clarification=True),
            service=_FakeService(last_request=None, last_response=None),
        )
        turn = _capture_turn("mental health 98101", agent, "Could you clarify?")
        self.assertFalse(turn.searched)
        self.assertEqual(turn.request_specialties, [])
        self.assertEqual(turn.provider_states, [])
        self.assertEqual(turn.provider_count, 0)
        self.assertFalse(turn.html_has_card)
        self.assertTrue(turn.parsed_needs_clarification)
        self.assertFalse(turn.emergency_routed)

    def test_capture_search_turn(self):
        from eval.trace import _capture_turn
        from provider_search.models import ProviderSearchRequest

        request = ProviderSearchRequest(
            specialties=("cardiology",), preferred_languages=("spanish",)
        )
        response = _FakeResponse([_FakeResult("CA"), _FakeResult("tx")])
        agent = _CaptureAgent(
            parsed=_FakeParsed(specialties=["cardiology"]),
            service=_FakeService(last_request=request, last_response=response),
            last_navigation_mode="emergency",
        )
        turn = _capture_turn("cardiology 98101", agent, "<div class='provider-card'>")
        self.assertTrue(turn.searched)
        self.assertEqual(turn.request_specialties, ["cardiology"])
        self.assertEqual(turn.request_preferred_languages, ["spanish"])
        self.assertEqual(turn.provider_states, ["CA", "TX"])
        self.assertEqual(turn.provider_count, 2)
        self.assertTrue(turn.html_has_card)
        self.assertTrue(turn.emergency_routed)

    def test_provider_state_falls_back_to_address(self):
        from eval.trace import _provider_state

        # ClinicalTables leaves .state blank; state must be parsed from the address.
        blank = _FakeProvider(state=None, address="710 LAWRENCE EXPY, SANTA CLARA, CA 98101")
        self.assertEqual(_provider_state(blank), "CA")
        # A populated structured field wins and is upper-cased.
        structured = _FakeProvider(state="ny", address=None)
        self.assertEqual(_provider_state(structured), "NY")
        # No usable location at all -> empty string.
        self.assertEqual(_provider_state(_FakeProvider(state=None, address=None)), "")

    def test_errored_run_is_not_cached(self):
        from eval.dataset import GoldLabels, LanguageVariant, Scenario

        scenario = Scenario(
            id="t-err",
            category="c",
            dimension="d",
            gold=GoldLabels(None, None, False, False, False, None),
            variants={
                "en": LanguageVariant("en", ["hi"], "english_seed", "human_verified", None, None)
            },
        )
        with tempfile.TemporaryDirectory() as d:
            trace = run_trace(
                scenario,
                "en",
                _RaisingAgent(),
                client=None,
                settings={"max_tokens": 1, "temperature": 0.0, "top_p": 1.0},
                cache_dir=d,
            )
            self.assertIsNotNone(trace.error)
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()
