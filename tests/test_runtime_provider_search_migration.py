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


class _SequencedChatClient:
    def __init__(self) -> None:
        self.calls = []

    def chat_completion(self, messages, max_tokens, temperature, top_p):
        self.calls.append(messages)
        return type("Completion", (), {"choices": []})()


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


class CareLocatorAgentProviderSearchRuntimeTests(unittest.TestCase):
    def test_handle_request_uses_provider_search_service_for_provider_cards(self) -> None:
        provider = build_canonical_provider(
            provider_id="1619271780",
            name="Harmony Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="San Francisco",
            state="CA",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            insurance_reported=("Medicare", "Aetna"),
            phone="415-555-0100",
        )
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("Primary Care",),
                location="San Francisco, CA",
                insurance=("Medicare",),
                preferred_languages=("English",),
            ),
            provider_results=(
                ProviderSearchResult(
                    provider=provider,
                    score=7.0,
                    source=provider.source,
                    retriever_metadata={"ranking_version": "deterministic-v1"},
                ),
            ),
            search_trace=SearchTrace(
                sources_attempted=("clinicaltables:npi_idv", "clinicaltables:npi_org"),
                sources_used=("NPI Registry (individual)",),
                cache_hit=False,
                total_candidates=1,
            ),
        )

        agent = CareLocatorAgent(provider_search_service=service)

        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care in San Francisco",
            medical_need=True,
            location="San Francisco, CA",
            specialties=["Primary Care"],
            insurance=["Medicare"],
            preferred_languages=["English"],
            keywords=[],
            patient_context=None,
        )

        client = _SequencedChatClient()

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                client,
                "primary care 94110",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        service.search.assert_called_once_with(
            ProviderSearchRequest(
                specialties=("Primary Care",),
                location="San Francisco, CA",
                insurance=("Medicare",),
                preferred_languages=("English",),
                keywords=(),
            ),
            limit=agent.ctss_max_results,
        )
        self.assertEqual(len(client.calls), 0)
        self.assertIn('<div class="provider-card__title">1. Harmony Family Clinic</div>', result)
        self.assertIn('Source</span><span class="provider-card__meta-value">NPI Registry (individual)</span>', result)
        self.assertIn('Listed insurance</span><span class="provider-card__value">Medicare, Aetna (reported only; network participation is not verified here)</span>', result)

    def test_default_init_works_without_cache_path_or_legacy_repository_dependency(self) -> None:
        with patch("provider_search.cache.resolve_provider_cache_path", return_value=None):
            agent = CareLocatorAgent()

        self.assertIsNotNone(agent.provider_search_service)
        self.assertFalse(hasattr(agent, "provider_repository"))
        self.assertFalse(agent.provider_search_service.cache.enabled)

    def test_handle_request_retries_malformed_interpret_json_then_falls_back_for_missing_final_content(self) -> None:
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(),
            search_trace=SearchTrace(),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {
                    "content": "not valid json",
                    "finish_reason": "stop",
                },
                {
                    "content": (
                        '{"detected_language":"English","response_language":"English","summary":"find a doctor",'
                        '"medical_need":true,"location":null,"specialties":[],"insurance":[],'
                        '"preferred_languages":[],"keywords":[],"patient_context":null,'
                        '"care_setting":null,"urgency":null,"needs_clarification":false,'
                        '"follow_up_focus":[]}'
                    ),
                    "finish_reason": "stop",
                },
                {
                    "content": None,
                    "finish_reason": "length",
                },
            ]
        )

        result = agent.handle_request(
            client,
            "find a doctor near me",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertIn("What city and state or ZIP code should I search?", result)
        self.assertIn(
            "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?",
            result,
        )
        self.assertIn("Important safety and trust notes:", result)
        self.assertEqual(len(client.calls), 3)
        service.search.assert_not_called()

    def test_handle_request_uses_emergency_fallback_when_final_model_response_is_truncated(self) -> None:
        service = Mock()
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="chest pain",
            medical_need=True,
            location="San Francisco, CA",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )
        client = _ScriptedChatClient(
            [
                {
                    "content": "Truncated partial response",
                    "finish_reason": "length",
                }
            ]
        )

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                client,
                "I have chest pain and trouble breathing",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertNotIn("Truncated partial response", result)
        self.assertIn(
            "If symptoms are severe or life-threatening, call emergency services now or go to the nearest emergency room.",
            result,
        )
        self.assertIn("Important safety and trust notes:", result)
        self.assertEqual(len(client.calls), 1)
        service.search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
