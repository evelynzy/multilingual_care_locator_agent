import sys
import types
import unittest
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

        with patch.object(
            CareLocatorAgent,
            "_initialize_clinicaltables_field_maps",
            return_value=None,
        ):
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

    def test_default_init_works_without_cache_path_or_local_repository(self) -> None:
        with patch("provider_search.cache.resolve_provider_cache_path", return_value=None):
            agent = CareLocatorAgent()

        self.assertIsNotNone(agent.provider_search_service)
        self.assertIsNone(agent.provider_repository)
        self.assertFalse(agent.provider_search_service.cache.enabled)


if __name__ == "__main__":
    unittest.main()
