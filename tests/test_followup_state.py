"""Deterministic multi-turn follow-up state tests for ``CareLocatorAgent``.

These are characterization tests for the highest-risk area: how parsed-intent
state is threaded and merged across conversation turns inside
``CareLocatorAgent.handle_request`` (and its ``_merge_parsed_queries`` helper).

The harness is copied verbatim from
``tests/test_runtime_provider_search_migration.py`` so the two files behave
identically under any test ordering:

* the module-level ``huggingface_hub`` / ``requests`` stub block (a stubbed
  ``requests`` without ``.exceptions`` may be installed under full-suite
  ordering, so we never touch ``requests.exceptions``); and
* the small ``_ScriptedChatClient`` helper.

Each test drives **two** real ``handle_request`` calls with a running
``history`` list. We patch ``_interpret_user_need`` with a ``side_effect`` list
to supply the parsed intent for each interpret invocation -- exactly as the
existing ``test_handle_request_history_merge_*`` tests do. Note that when
``history`` is non-empty, ``handle_request`` calls ``_interpret_user_need``
twice (once with full history -> "full", once with an empty history ->
"latest") before merging, so turn 2 consumes two side-effect entries.
"""

import json
import sys
import types
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch


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


from care_agent import CareLocatorAgent, ParsedCareQuery
from provider_search.models import (
    ProviderSearchRequest,
    ProviderSearchResponse,
    ProviderSearchResult,
    SearchTrace,
)
from provider_search.normalization import build_canonical_provider


class _ScriptedChatClient:
    def __init__(self, responses: List[Dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls = []

    def chat_completion(self, messages, max_tokens, temperature, top_p):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        response_index = min(len(self.calls) - 1, len(self.responses) - 1)
        response = self.responses[response_index]
        if not response.get("include_choice", True):
            return type("Completion", (), {"choices": []})()

        message_payload: Dict[str, Optional[str]] = {}
        if "content" in response:
            message_payload["content"] = response.get("content")

        choice = type(
            "Choice",
            (),
            {
                "message": message_payload,
                "finish_reason": response.get("finish_reason", "stop"),
            },
        )()
        return type("Completion", (), {"choices": [choice]})()


_CARE_NEED_QUESTION = (
    "What kind of care do you need (for example primary care, pediatrics, "
    "dermatology, ENT, or urgent care)?"
)
_LOCATION_QUESTION = "What city and state or ZIP code should I search?"


def _provider_search_response_with(provider) -> ProviderSearchResponse:
    return ProviderSearchResponse(
        request=ProviderSearchRequest(),
        provider_results=(
            ProviderSearchResult(
                provider=provider,
                score=1.0,
                source="provider_search_service",
            ),
        ),
        search_trace=SearchTrace(),
    )


class CareLocatorAgentFollowUpStateTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # 1. child + allergy ambiguity asked in turn 1, resolved in turn 2.
    # ------------------------------------------------------------------
    def test_child_allergy_ambiguity_resolved_in_later_turn_uses_pediatrics(
        self,
    ) -> None:
        pediatrics_provider = build_canonical_provider(
            provider_id="provider-pediatrics-followup",
            name="Santa Clara Pediatrics",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Pediatrics",
            specialties=("Pediatrics",),
            phone="408-555-0140",
        )
        service = Mock()
        service.search.return_value = _provider_search_response_with(
            pediatrics_provider
        )
        agent = CareLocatorAgent(provider_search_service=service)

        # Turn 1: ambiguous "child + allergies" intent -> agent abstains from a
        # specialty and asks for clarification instead of searching.
        turn1_query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="doctor for my child and allergies 98101",
            medical_need=True,
            location="98101",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=True,
            follow_up_focus=["specialty clarification"],
        )
        # Turn 2 full-history interpret: still reflects the unresolved
        # clarification state carried over from turn 1.
        turn2_full = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="child and allergies pediatrics 98101",
            medical_need=True,
            location="98101",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=True,
            follow_up_focus=["specialty clarification"],
        )
        # Turn 2 latest-only interpret: the concrete answer "Pediatrics".
        turn2_latest = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="Pediatrics",
            medical_need=True,
            location=None,
            specialties=["Pediatrics"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting=None,
            needs_clarification=False,
            follow_up_focus=[],
        )

        client = _ScriptedChatClient([{"content": "ok"}])

        with patch.object(
            agent,
            "_interpret_user_need",
            side_effect=[turn1_query, turn2_full, turn2_latest],
        ), patch.object(
            agent,
            "_compose_response",
            side_effect=lambda *_args, **_kwargs: json.dumps(_args[1]),
        ), patch.object(
            agent,
            "_compose_result_card_response",
            side_effect=lambda payload: json.dumps(payload["query"]),
        ):
            turn1_result = agent.handle_request(
                client,
                "doctor for my child and allergies 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

            # Turn 1 asks the care-need clarifying question and does NOT search.
            self.assertIn(_CARE_NEED_QUESTION, turn1_result)
            service.search.assert_not_called()

            history = [
                {
                    "role": "user",
                    "content": "doctor for my child and allergies 98101",
                },
                {"role": "assistant", "content": turn1_result},
            ]

            turn2_result = agent.handle_request(
                client,
                "Pediatrics",
                history,
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        # Turn 2 now resolves to Pediatrics: it searches (does not re-ask) and
        # carries the specialty plus the location remembered from turn 1.
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("Pediatrics",))
        self.assertEqual(provider_request.location, "98101")

        merged_query = json.loads(turn2_result)
        self.assertEqual(merged_query["specialties"], ["Pediatrics"])
        self.assertNotIn("specialty clarification", merged_query["follow_up_focus"])
        self.assertFalse(merged_query["needs_clarification"])

    # ------------------------------------------------------------------
    # 2. specialty preserved across a location-only follow-up turn.
    # ------------------------------------------------------------------
    def test_specialty_preserved_across_location_only_followup(self) -> None:
        dermatology_provider = build_canonical_provider(
            provider_id="provider-dermatology-followup",
            name="Mission Dermatology",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="San Francisco",
            state="CA",
            taxonomy="Dermatology",
            specialties=("Dermatology",),
            phone="415-555-0160",
        )
        service = Mock()
        service.search.return_value = _provider_search_response_with(
            dermatology_provider
        )
        agent = CareLocatorAgent(provider_search_service=service)

        # Turn 1: clear specialty, no location -> agent asks for a location.
        turn1_query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="dermatology",
            medical_need=True,
            location=None,
            specialties=["Dermatology"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=False,
            follow_up_focus=[],
        )
        # Turn 2 full-history interpret: remembers dermatology, now with a ZIP.
        turn2_full = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="dermatology 94110",
            medical_need=True,
            location="94110",
            specialties=["Dermatology"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=False,
            follow_up_focus=[],
        )
        # Turn 2 latest-only interpret: a bare ZIP, no specialty in this turn.
        turn2_latest = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="94110",
            medical_need=True,
            location="94110",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting=None,
            needs_clarification=False,
            follow_up_focus=[],
        )

        client = _ScriptedChatClient([{"content": "ok"}])

        with patch.object(
            agent,
            "_interpret_user_need",
            side_effect=[turn1_query, turn2_full, turn2_latest],
        ), patch.object(
            agent,
            "_compose_response",
            side_effect=lambda *_args, **_kwargs: json.dumps(_args[1]),
        ), patch.object(
            agent,
            "_compose_result_card_response",
            side_effect=lambda payload: json.dumps(payload["query"]),
        ):
            turn1_result = agent.handle_request(
                client,
                "dermatology",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

            # Turn 1 asks only for a location and does NOT search.
            self.assertIn(_LOCATION_QUESTION, turn1_result)
            self.assertNotIn(_CARE_NEED_QUESTION, turn1_result)
            service.search.assert_not_called()

            history = [
                {"role": "user", "content": "dermatology"},
                {"role": "assistant", "content": turn1_result},
            ]

            turn2_result = agent.handle_request(
                client,
                "94110",
                history,
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        # The location-only follow-up must not lose the turn-1 specialty.
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("Dermatology",))
        self.assertEqual(provider_request.location, "94110")

        merged_query = json.loads(turn2_result)
        self.assertEqual(merged_query["specialties"], ["Dermatology"])
        self.assertEqual(merged_query["location"], "94110")

    # ------------------------------------------------------------------
    # 3. a clear intent switch drops the stale specialty.
    # ------------------------------------------------------------------
    def test_intent_switch_drops_stale_specialty(self) -> None:
        cardiology_provider = build_canonical_provider(
            provider_id="provider-cardiology-followup",
            name="South Bay Cardiology",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Cardiology",
            specialties=("Cardiology",),
            phone="408-555-0170",
        )
        service = Mock()
        service.search.return_value = _provider_search_response_with(
            cardiology_provider
        )
        agent = CareLocatorAgent(provider_search_service=service)

        # Turn 1: cardiology with a ZIP -> a real search runs.
        turn1_query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="cardiology 98101",
            medical_need=True,
            location="98101",
            specialties=["Cardiology"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=False,
            follow_up_focus=[],
        )
        # Turn 2 full-history interpret: pessimistically still anchored on the
        # stale cardiology intent from turn 1.
        turn2_full = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="cardiology dermatologist 98101",
            medical_need=True,
            location="98101",
            specialties=["Cardiology"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=False,
            follow_up_focus=[],
        )
        # Turn 2 latest-only interpret: the explicit switch to dermatology.
        turn2_latest = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="actually I need a dermatologist in 98101",
            medical_need=True,
            location="98101",
            specialties=["Dermatology"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
            care_setting="specialist",
            needs_clarification=False,
            follow_up_focus=[],
        )

        client = _ScriptedChatClient([{"content": "ok"}])

        with patch.object(
            agent,
            "_interpret_user_need",
            side_effect=[turn1_query, turn2_full, turn2_latest],
        ), patch.object(
            agent,
            "_compose_result_card_response",
            side_effect=lambda payload: json.dumps(payload["query"]),
        ):
            turn1_result = agent.handle_request(
                client,
                "cardiology 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

            # Turn 1 searched for cardiology.
            service.search.assert_called_once()
            self.assertEqual(
                service.search.call_args[0][0].specialties, ("Cardiology",)
            )
            service.search.reset_mock()

            history = [
                {"role": "user", "content": "cardiology 98101"},
                {"role": "assistant", "content": turn1_result},
            ]

            turn2_result = agent.handle_request(
                client,
                "actually I need a dermatologist in 98101",
                history,
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        # The new search must be dermatology-only; the stale cardiology intent
        # is dropped, not unioned in.
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("Dermatology",))
        self.assertNotIn("Cardiology", provider_request.specialties)
        self.assertEqual(provider_request.location, "98101")

        merged_query = json.loads(turn2_result)
        self.assertEqual(merged_query["specialties"], ["Dermatology"])

    # ------------------------------------------------------------------
    # Characterization of ``_trusted_resource_fallback`` region filtering.
    # ------------------------------------------------------------------
    def test_trusted_resource_fallback_region_filter_depends_on_extracted_location(
        self,
    ) -> None:
        agent = CareLocatorAgent(provider_search_service=Mock())

        us_resource = {
            "name": "US National Care Directory",
            "url": "https://example.org/us-directory",
            "description": "Nationwide trusted public directory.",
            "regions": ["united states"],
        }
        agent.fallback_resources = [us_resource]

        def _query(location: Optional[str]) -> ParsedCareQuery:
            return ParsedCareQuery(
                detected_language="English",
                response_language="English",
                summary="please help me find care",
                medical_need=True,
                location=location,
                specialties=[],
                insurance=[],
                preferred_languages=[],
                keywords=[],
                patient_context=None,
            )

        # With a US ZIP, region context resolves to the United States and the
        # region-filtered resource is surfaced.
        included = agent._trusted_resource_fallback(_query("94110"))
        self.assertEqual(
            [resource["name"] for resource in included],
            ["US National Care Directory"],
        )

        # Corrected behavior: when location is unknown (None), national/US-scoped
        # resources must still be surfaced rather than dropped.
        without_location = agent._trusted_resource_fallback(_query(None))
        self.assertEqual(
            [resource["name"] for resource in without_location],
            ["US National Care Directory"],
        )


if __name__ == "__main__":
    unittest.main()
