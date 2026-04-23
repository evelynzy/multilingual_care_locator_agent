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
    SourceSearchResult,
    SearchTrace,
    SourceTrace,
)
from provider_search.normalization import build_canonical_provider
from provider_search.service import ProviderSearchService


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


class _PediatricRetryClinicalTablesSource:
    def __init__(
        self,
        specialty_retry_providers: List[Any],
        *,
        location_only_providers: Optional[List[Any]] = None,
    ) -> None:
        self.specialty_retry_providers = list(specialty_retry_providers)
        self.location_only_providers = list(location_only_providers or [])
        self.calls = []

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if (
            request.search_terms == "Pediatrics"
            and request.query_filter == "addr_practice.state:NY AND addr_practice.zip:10013"
        ):
            return SourceSearchResult(
                providers=[],
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=0,
                ),
            )
        if (
            request.search_terms == "Pediatrics"
            and request.query_filter == "addr_practice.state:NY"
        ):
            return SourceSearchResult(
                providers=list(self.specialty_retry_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.specialty_retry_providers),
                ),
            )
        if request.search_terms == "Manhattan, NY 10013":
            return SourceSearchResult(
                providers=list(self.location_only_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.location_only_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class _NearbyDentalClinicalTablesSource:
    def __init__(
        self,
        local_zip_providers: List[Any],
        nearby_state_providers: List[Any],
    ) -> None:
        self.local_zip_providers = list(local_zip_providers)
        self.nearby_state_providers = list(nearby_state_providers)
        self.calls = []

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if (
            request.query_filter == "addr_practice.zip:33012"
            and request.search_terms in {"Dentistry", "Dentistry 33012"}
        ):
            return SourceSearchResult(
                providers=list(self.local_zip_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.local_zip_providers),
                ),
            )
        if (
            request.query_filter == "addr_practice.state:FL"
            and request.search_terms in {"Dentistry", "Dentistry FL"}
        ):
            return SourceSearchResult(
                providers=list(self.nearby_state_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.nearby_state_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class _KeywordOnlyClinicalTablesSource:
    def __init__(self) -> None:
        self.calls = []

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="NPI Registry (organization)",
            dataset=dataset,
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        ).with_updates(description="Child health imaging and pediatric scans.")
        return SourceSearchResult(
            providers=[provider],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=1,
            ),
        )


class _PrimaryCareRetryClinicalTablesSource:
    def __init__(
        self,
        retry_providers: List[Any],
        *,
        location_only_providers: Optional[List[Any]] = None,
    ) -> None:
        self.retry_providers = list(retry_providers)
        self.location_only_providers = list(location_only_providers or [])
        self.calls = []

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if (
            request.search_terms == "Primary Care"
            and request.query_filter == "addr_practice.state:TX AND addr_practice.zip:75001"
        ):
            return SourceSearchResult(
                providers=[],
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=0,
                ),
            )
        if (
            request.search_terms == "Primary Care"
            and request.query_filter == "addr_practice.state:TX"
        ):
            return SourceSearchResult(
                providers=list(self.retry_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.retry_providers),
                ),
            )
        if request.search_terms == "Dallas, TX 75001":
            return SourceSearchResult(
                providers=list(self.location_only_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.location_only_providers),
                ),
            )
        return SourceSearchResult(
            providers=[],
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=0,
            ),
        )


class _DuplicatePrimaryCareClinicalTablesSource:
    def __init__(self) -> None:
        self.calls = []

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if dataset == "npi_idv":
            providers = [
                build_canonical_provider(
                    provider_id="1111111111",
                    name="Dallas Family Clinic",
                    source_name="NPI Registry (individual)",
                    dataset=dataset,
                    address="123 Main St",
                    city="Dallas",
                    state="TX",
                    taxonomy="Primary Care",
                    specialties=("Primary Care",),
                    phone="214-555-0100",
                )
            ]
        elif dataset == "npi_org":
            providers = [
                build_canonical_provider(
                    provider_id="2222222222",
                    name="Dallas Family Clinic",
                    source_name="NPI Registry (organization)",
                    dataset=dataset,
                    address="123 Main St",
                    city="Dallas",
                    state="TX",
                    taxonomy="Primary Care",
                    specialties=("Primary Care",),
                    phone="214-555-0100",
                )
            ]
        else:
            providers = []
        return SourceSearchResult(
            providers=providers,
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=len(providers),
            ),
        )


class _DifferentServiceLineClinicalTablesSource:
    def __init__(self) -> None:
        self.calls = []

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if dataset == "npi_idv":
            providers = [
                build_canonical_provider(
                    provider_id="1111111111",
                    name="Dallas Family Clinic",
                    source_name="NPI Registry (organization)",
                    dataset=dataset,
                    address="123 Main St",
                    city="Dallas",
                    state="TX",
                    taxonomy="Primary Care",
                    specialties=("Primary Care",),
                    phone="214-555-0100",
                )
            ]
        elif dataset == "npi_org":
            providers = [
                build_canonical_provider(
                    provider_id="2222222222",
                    name="Dallas Family Clinic",
                    source_name="NPI Registry (organization)",
                    dataset=dataset,
                    address="123 Main St",
                    city="Dallas",
                    state="TX",
                    taxonomy="Pediatrics",
                    specialties=("Pediatrics",),
                    phone="214-555-0100",
                )
            ]
        else:
            providers = []
        return SourceSearchResult(
            providers=providers,
            trace=SourceTrace(
                source_name="clinicaltables",
                dataset=dataset,
                status_code=200,
                result_count=len(providers),
            ),
        )


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

    def test_handle_request_logs_opt_in_final_visible_result_counts(self) -> None:
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("Primary Care",),
                location="San Francisco, CA",
            ),
            provider_results=(),
            search_trace=SearchTrace(
                cache_hit=False,
                request_fingerprint="debug-fingerprint",
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
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )
        trusted_fallback = [
            {
                "name": "Medicare Care Compare",
                "location": "San Francisco, CA",
                "website": "https://www.medicare.gov/care-compare/",
                "description": "Official U.S. tool to compare Medicare-enrolled providers.",
                "source": "Trusted public directories",
            }
        ]

        with patch.dict("os.environ", {"PROVIDER_SEARCH_DEBUG": "1"}):
            with patch.object(agent, "_interpret_user_need", return_value=query), patch.object(
                agent,
                "_trusted_resource_fallback",
                return_value=trusted_fallback,
            ):
                with self.assertLogs("care_agent", level="INFO") as captured:
                    agent.handle_request(
                        _SequencedChatClient(),
                        "primary care 94110",
                        [],
                        max_tokens=256,
                        temperature=0.2,
                        top_p=0.9,
                    )

        joined_logs = "\n".join(captured.output)
        self.assertIn("care_agent_result_debug request_fingerprint=debug-fingerprint", joined_logs)
        self.assertIn("local_results=0", joined_logs)
        self.assertIn("fallback_results=1", joined_logs)
        self.assertIn("final_visible=1", joined_logs)

    def test_default_init_works_without_cache_path_or_legacy_repository_dependency(self) -> None:
        with patch("provider_search.cache.resolve_provider_cache_path", return_value=None):
            agent = CareLocatorAgent()

        self.assertIsNotNone(agent.provider_search_service)
        self.assertFalse(hasattr(agent, "provider_repository"))
        self.assertFalse(agent.provider_search_service.cache.enabled)

    def test_handle_request_primary_care_75001_renders_single_card_for_duplicate_org_pair(self) -> None:
        service = ProviderSearchService(
            clinicaltables_source=_DuplicatePrimaryCareClinicalTablesSource(),
            cache=None,
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care 75001",
            medical_need=True,
            location="Dallas, TX 75001",
            specialties=["Primary Care"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )
        client = _SequencedChatClient()

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                client,
                "primary care 75001",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result.count("provider-card__title"), 1)
        self.assertIn('<div class="provider-card__title">1. Dallas Family Clinic</div>', result)
        self.assertNotIn("2. Dallas Family Clinic", result)

    def test_handle_request_primary_care_75001_uses_relaxed_retry_and_avoids_fallback_when_provider_found(self) -> None:
        primary_care_provider = build_canonical_provider(
            provider_id="provider-primary-care",
            name="Dallas Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        location_only_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Dallas",
            state="TX",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
            phone="214-555-0199",
        )
        source = _PrimaryCareRetryClinicalTablesSource(
            [primary_care_provider],
            location_only_providers=[location_only_provider],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care 75001",
            medical_need=True,
            location="Dallas, TX 75001",
            specialties=["Primary Care"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query), patch.object(
            agent,
            "_trusted_resource_fallback",
        ) as trusted_fallback:
            result = agent.handle_request(
                _SequencedChatClient(),
                "primary care 75001",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(source.calls), 4)
        _, first_request = source.calls[0]
        _, second_request = source.calls[1]
        _, retry_request = source.calls[2]
        self.assertEqual(first_request.search_terms, "Primary Care")
        self.assertEqual(second_request.search_terms, "Primary Care Dallas TX 75001")
        self.assertEqual(retry_request.search_terms, "Primary Care")
        self.assertEqual(
            first_request.query_filter,
            "addr_practice.state:TX AND addr_practice.zip:75001",
        )
        self.assertEqual(retry_request.query_filter, "addr_practice.state:TX")
        self.assertNotIn("Dallas, TX 75001", [request.search_terms for _, request in source.calls])
        trusted_fallback.assert_not_called()
        self.assertIn("Dallas Family Clinic", result)
        self.assertNotIn("Downtown Imaging Associates", result)
        self.assertNotIn("Medicare Care Compare", result)

    def test_handle_request_primary_care_75001_renders_two_cards_when_duplicate_pair_crowds_retry_limit(self) -> None:
        duplicate_primary_care_individual = build_canonical_provider(
            provider_id="provider-primary-care-individual",
            name="Dallas Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        duplicate_primary_care_org = build_canonical_provider(
            provider_id="provider-primary-care-org",
            name="Dallas Family Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="123 Main St",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        second_visible_provider = build_canonical_provider(
            provider_id="provider-second-visible",
            name="Zzz Addison Primary Care",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            address="456 Belt Line Rd",
            city="Addison",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="972-555-0199",
        )
        source = _PrimaryCareRetryClinicalTablesSource(
            [
                duplicate_primary_care_individual,
                duplicate_primary_care_org,
                second_visible_provider,
            ]
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv", "npi_org"),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care 75001",
            medical_need=True,
            location="Dallas, TX 75001",
            specialties=["Primary Care"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                _SequencedChatClient(),
                "primary care 75001",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(source.calls), 8)
        self.assertEqual(result.count("provider-card__title"), 2)
        self.assertIn('<div class="provider-card__title">1. Dallas Family Clinic</div>', result)
        self.assertIn('<div class="provider-card__title">2. Zzz Addison Primary Care</div>', result)

    def test_handle_request_same_site_different_service_lines_renders_separate_cards(self) -> None:
        service = ProviderSearchService(
            clinicaltables_source=_DifferentServiceLineClinicalTablesSource(),
            cache=None,
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="care 75001",
            medical_need=True,
            location="Dallas, TX 75001",
            specialties=[],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query):
            with patch.object(
                agent,
                "_build_navigation_guidance",
                return_value={
                    "mode": "search",
                    "care_setting_guidance": None,
                    "follow_up_questions": [],
                    "specialist_plan_guidance": None,
                    "location_only": False,
                },
            ):
                result = agent.handle_request(
                    _SequencedChatClient(),
                    "care 75001",
                    [],
                    max_tokens=256,
                    temperature=0.2,
                    top_p=0.9,
                )

        self.assertEqual(result.count("provider-card__title"), 2)
        self.assertIn('<div class="provider-card__title">1. Dallas Family Clinic</div>', result)
        self.assertIn('<div class="provider-card__title">2. Dallas Family Clinic</div>', result)
        self.assertIn("Primary Care", result)
        self.assertIn("Pediatrics", result)

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

    def test_handle_request_rescues_primary_care_zip_from_malformed_interpret_json(self) -> None:
        primary_care_provider = build_canonical_provider(
            provider_id="provider-primary-care",
            name="Dallas Family Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Dallas",
            state="TX",
            taxonomy="Primary Care",
            specialties=("Primary Care",),
            phone="214-555-0100",
        )
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("Primary Care",),
                location="75001",
            ),
            provider_results=(
                ProviderSearchResult(
                    provider=primary_care_provider,
                    score=1.0,
                    source="provider_search_service",
                ),
            ),
            search_trace=SearchTrace(
                source_traces=(
                    SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_idv",
                        status_code=200,
                        result_count=1,
                    ),
                ),
                sources_attempted=("clinicaltables",),
                sources_used=("clinicaltables",),
                total_candidates=1,
            ),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {"content": "not valid json", "finish_reason": "stop"},
                {"content": "{still not valid json", "finish_reason": "stop"},
            ]
        )

        result = agent.handle_request(
            client,
            "primary care 75001",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("Primary Care",))
        self.assertEqual(provider_request.location, "75001")
        self.assertIn("Dallas Family Clinic", result)
        self.assertIn("Primary Care", result)

    def test_handle_request_rescue_does_not_misread_pcp_in_dallas_tx_as_pcp_in(self) -> None:
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Dallas, TX",
            ),
            search_trace=SearchTrace(),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {"content": "not valid json", "finish_reason": "stop"},
                {"content": "{still not valid json", "finish_reason": "stop"},
            ]
        )

        result = agent.handle_request(
            client,
            "need PCP in Dallas TX",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("Primary Care",))
        self.assertEqual(provider_request.location, "Dallas, TX")
        self.assertNotIn("PCP, IN", result)

    def test_handle_request_rescues_common_specialty_from_malformed_interpret_json(self) -> None:
        ent_provider = build_canonical_provider(
            provider_id="provider-ent",
            name="Austin ENT Clinic",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Austin",
            state="TX",
            taxonomy="ENT",
            specialties=("ENT",),
            phone="512-555-0100",
        )
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("ENT",),
                location="Austin, TX",
            ),
            provider_results=(
                ProviderSearchResult(
                    provider=ent_provider,
                    score=1.0,
                    source="provider_search_service",
                ),
            ),
            search_trace=SearchTrace(
                source_traces=(
                    SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_idv",
                        status_code=200,
                        result_count=1,
                    ),
                ),
                sources_attempted=("clinicaltables",),
                sources_used=("clinicaltables",),
                total_candidates=1,
            ),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {"content": "not valid json", "finish_reason": "stop"},
                {"content": '{"invalid_json":', "finish_reason": "stop"},
            ]
        )

        result = agent.handle_request(
            client,
            "need an ENT near Austin TX",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("ENT",))
        self.assertEqual(provider_request.location, "Austin, TX")
        self.assertIn("Austin ENT Clinic", result)
        self.assertNotIn(
            "What kind of care do you need (for example primary care, pediatrics, dermatology, ENT, or urgent care)?",
            result,
        )

    def test_handle_request_rescues_dentista_zip_from_malformed_interpret_json(self) -> None:
        local_zip_providers = [
            build_canonical_provider(
                provider_id="provider-local-1",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Pediatric Dentistry", "1223P0221X"),
            ),
            build_canonical_provider(
                provider_id="provider-local-2",
                name="Hialeah Square Dentistry, PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, General Practice", "1223G0001X"),
            ),
            build_canonical_provider(
                provider_id="provider-local-3",
                name="Caplin and Gober Dentistry, PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Periodontics", "1223P0300X"),
            ),
        ]
        nearby_state_provider = build_canonical_provider(
            provider_id="provider-nearby-1",
            name="Miami Lakes Dentistry Center",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Miami Lakes",
            state="FL",
            taxonomy="Dentistry",
            specialties=("Dentistry",),
            phone="305-555-0101",
        )
        source = _NearbyDentalClinicalTablesSource(local_zip_providers, [nearby_state_provider])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {"content": "not valid json", "finish_reason": "stop"},
                {"content": '{"invalid_json":', "finish_reason": "stop"},
            ]
        )

        result = agent.handle_request(
            client,
            "dentista 33012",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(len(source.calls), 4)
        _, first_request = source.calls[0]
        _, retry_request = source.calls[2]
        self.assertEqual(first_request.search_terms, "Dentistry")
        self.assertEqual(first_request.query_filter, "addr_practice.zip:33012")
        self.assertEqual(retry_request.search_terms, "Dentistry")
        self.assertEqual(retry_request.query_filter, "addr_practice.state:FL")
        self.assertIn("Miami Lakes Dentistry Center", result)

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

    def test_handle_request_retries_when_initial_interpret_content_is_none(self) -> None:
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(),
            search_trace=SearchTrace(),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {
                    "content": None,
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
                    "content": "Clarification answer.",
                    "finish_reason": "stop",
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

        self.assertIn("Clarification answer.", result)
        self.assertIn("Important safety and trust notes:", result)
        self.assertEqual(len(client.calls), 3)
        service.search.assert_not_called()

    def test_handle_request_uses_default_query_when_retry_interpret_content_is_none(self) -> None:
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
                    "content": None,
                    "finish_reason": "stop",
                },
                {
                    "content": "Clarification answer.",
                    "finish_reason": "stop",
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

        self.assertIn("Clarification answer.", result)
        self.assertIn("Important safety and trust notes:", result)
        self.assertEqual(len(client.calls), 3)
        service.search.assert_not_called()

    def test_handle_request_uses_us_trusted_fallback_resources_for_city_state_zero_results(self) -> None:
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Pittsburgh, PA",
            ),
            provider_results=(),
            fallback_resources=(),
            missing_location_hint=None,
            search_trace=SearchTrace(
                source_traces=(
                    SourceTrace(source_name="clinicaltables", dataset="npi_idv", result_count=0),
                    SourceTrace(source_name="clinicaltables", dataset="npi_org", result_count=0),
                ),
                sources_attempted=("clinicaltables:npi_idv", "clinicaltables:npi_org"),
                total_candidates=0,
            ),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care in Pittsburgh",
            medical_need=True,
            location="Pittsburgh, PA",
            specialties=["Primary Care"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )
        captured: Dict[str, Any] = {}

        def _capture(payload: Dict[str, Any]) -> str:
            captured["payload"] = payload
            return "fallback-cards"

        with patch.object(agent, "_interpret_user_need", return_value=query), patch.object(
            agent,
            "_compose_result_card_response",
            side_effect=_capture,
        ):
            result = agent.handle_request(
                _SequencedChatClient(),
                "primary care in Pittsburgh, PA",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "fallback-cards")
        self.assertEqual(captured["payload"]["local_results"], [])
        self.assertEqual(len(captured["payload"]["fallback_results"]), 1)
        self.assertEqual(
            captured["payload"]["fallback_results"][0]["name"],
            "Medicare Care Compare",
        )
        self.assertNotIn("notes", captured["payload"])

    def test_handle_request_uses_source_failure_note_instead_of_no_providers_note(self) -> None:
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("Primary Care",),
                location="Pittsburgh, PA",
            ),
            provider_results=(),
            fallback_resources=(),
            missing_location_hint=None,
            search_trace=SearchTrace(
                source_traces=(
                    SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_idv",
                        result_count=0,
                        error="clinicaltables timeout",
                    ),
                    SourceTrace(
                        source_name="clinicaltables",
                        dataset="npi_org",
                        result_count=0,
                        error="clinicaltables timeout",
                    ),
                ),
                sources_attempted=("clinicaltables:npi_idv", "clinicaltables:npi_org"),
                total_candidates=0,
            ),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="primary care in Pittsburgh",
            medical_need=True,
            location="Pittsburgh, PA",
            specialties=["Primary Care"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )
        captured: Dict[str, Any] = {}

        def _capture(payload: Dict[str, Any]) -> str:
            captured["payload"] = payload
            return "fallback-cards"

        with patch.object(agent, "_interpret_user_need", return_value=query), patch.object(
            agent,
            "_compose_result_card_response",
            side_effect=_capture,
        ):
            result = agent.handle_request(
                _SequencedChatClient(),
                "primary care in Pittsburgh, PA",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(result, "fallback-cards")
        self.assertEqual(
            captured["payload"]["notes"],
            "Provider search sources were temporarily unavailable. Showing trusted fallback resources when available.",
        )
        self.assertEqual(len(captured["payload"]["fallback_results"]), 1)
        self.assertNotEqual(
            captured["payload"]["notes"],
            "No providers were found via the configured provider search sources.",
        )

    def test_handle_request_pediatric_10013_keeps_only_provider_with_pediatric_evidence(self) -> None:
        pediatric_provider = build_canonical_provider(
            provider_id="provider-peds",
            name="Canal Pediatrics",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Pediatrics",
            specialties=("Pediatrics",),
            phone="212-555-0101",
        ).with_updates(description="Pediatric and child health visits.")
        radiology_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
            phone="212-555-0199",
        )
        location_only_providers = [
            radiology_provider.with_updates(provider_id=f"provider-radiology-{index}")
            for index in range(15)
        ]
        source = _PediatricRetryClinicalTablesSource(
            [pediatric_provider, radiology_provider],
            location_only_providers=location_only_providers,
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="中文",
            response_language="中文",
            summary="儿科10013",
            medical_need=True,
            location="Manhattan, NY 10013",
            specialties=["Pediatrics"],
            insurance=[],
            preferred_languages=[],
            keywords=["pediatric", "child health"],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                _SequencedChatClient(),
                "儿科 10013 曼哈顿",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(source.calls), 4)
        _, first_request = source.calls[0]
        _, second_request = source.calls[1]
        _, retry_request = source.calls[2]
        self.assertEqual(first_request.search_terms, "Pediatrics")
        self.assertEqual(second_request.search_terms, "Pediatrics Manhattan NY 10013")
        self.assertEqual(retry_request.search_terms, first_request.search_terms)
        self.assertEqual(
            first_request.query_filter,
            "addr_practice.state:NY AND addr_practice.zip:10013",
        )
        self.assertEqual(retry_request.query_filter, "addr_practice.state:NY")
        self.assertIn("Canal Pediatrics", result)
        self.assertNotIn("Downtown Imaging Associates", result)
        self.assertIn("Pediatrics", result)

    def test_handle_request_specialty_mismatch_keyword_match_does_not_bypass_specialty_gate(self) -> None:
        service = ProviderSearchService(
            clinicaltables_source=_KeywordOnlyClinicalTablesSource(),
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="中文",
            response_language="中文",
            summary="儿科10013",
            medical_need=True,
            location="Manhattan, NY 10013",
            specialties=["Pediatrics"],
            insurance=[],
            preferred_languages=[],
            keywords=["child health"],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                _SequencedChatClient(),
                "儿科 10013 曼哈顿",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertNotIn("Downtown Imaging Associates", result)
        self.assertIn("Medicare Care Compare", result)
        self.assertIn("Trusted public directories", result)

    def test_handle_request_pediatric_10013_keeps_live_org_candidate_with_nppes_pediatric_specialties(self) -> None:
        pediatric_org_provider = build_canonical_provider(
            provider_id="provider-org-peds",
            name="Downtown Kids Clinic",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Manhattan",
            state="NY",
            taxonomy="Clinic/Center",
            specialties=(
                "Clinic/Center",
                "Pediatrics",
                "208000000X",
                "Pediatric Gastroenterology",
                "2080P0206X",
            ),
            phone="212-555-0101",
        )
        source = _PediatricRetryClinicalTablesSource([pediatric_org_provider])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="中文",
            response_language="中文",
            summary="儿科10013",
            medical_need=True,
            location="Manhattan, NY 10013",
            specialties=["Pediatrics"],
            insurance=[],
            preferred_languages=[],
            keywords=["pediatric", "child health"],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                _SequencedChatClient(),
                "儿科 10013 曼哈顿",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(source.calls), 4)
        self.assertIn("Downtown Kids Clinic", result)
        self.assertNotIn("Medicare Care Compare", result)
        self.assertNotIn("Trusted public directories", result)

    def test_handle_request_dentista_33012_uses_same_state_nearby_retry_before_fallback(self) -> None:
        local_zip_providers = [
            build_canonical_provider(
                provider_id="provider-local-1",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Pediatric Dentistry", "1223P0221X"),
            ),
            build_canonical_provider(
                provider_id="provider-local-2",
                name="Hialeah Square Dentistry, PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, General Practice", "1223G0001X"),
            ),
            build_canonical_provider(
                provider_id="provider-local-3",
                name="Caplin and Gober Dentistry, PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Periodontics", "1223P0300X"),
            ),
        ]
        nearby_state_provider = build_canonical_provider(
            provider_id="provider-nearby-1",
            name="Miami Lakes Dentistry Center",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            city="Miami Lakes",
            state="FL",
            taxonomy="Dentistry",
            specialties=("Dentistry",),
            phone="305-555-0101",
        )
        source = _NearbyDentalClinicalTablesSource(local_zip_providers, [nearby_state_provider])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="Español",
            response_language="Español",
            summary="dentista 33012",
            medical_need=True,
            location="33012",
            specialties=["Dentistry"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query), patch.object(
            agent,
            "_trusted_resource_fallback",
        ) as trusted_fallback:
            result = agent.handle_request(
                _SequencedChatClient(),
                "dentista 33012",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(source.calls), 4)
        self.assertIn("Miami Lakes Dentistry Center", result)
        self.assertNotIn("Medicare Care Compare", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_dentista_33012_renders_two_cards_when_nearby_retry_has_duplicate_pair(self) -> None:
        local_zip_providers = [
            build_canonical_provider(
                provider_id="provider-local-1",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist",),
            ),
        ]
        duplicate_nearby_individual = build_canonical_provider(
            provider_id="provider-nearby-individual",
            name="Miami Lakes Dentistry Center",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            address="789 Oak Ave",
            city="Miami Lakes",
            state="FL",
            taxonomy="Dentistry",
            specialties=("Dentistry",),
            phone="305-555-0101",
        )
        duplicate_nearby_org = build_canonical_provider(
            provider_id="provider-nearby-org",
            name="Miami Lakes Dentistry Center",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="789 Oak Ave",
            city="Miami Lakes",
            state="FL",
            taxonomy="Dentistry",
            specialties=("Dentistry",),
            phone="305-555-0101",
        )
        second_visible_provider = build_canonical_provider(
            provider_id="provider-nearby-second",
            name="Zzz Family Dental",
            source_name="NPI Registry (organization)",
            dataset="npi_org",
            address="900 Pine St",
            city="Miami",
            state="FL",
            taxonomy="Dentistry",
            specialties=("Dentistry",),
            phone="305-555-0102",
        )
        source = _NearbyDentalClinicalTablesSource(
            local_zip_providers,
            [
                duplicate_nearby_individual,
                duplicate_nearby_org,
                second_visible_provider,
            ],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv", "npi_org"),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="Español",
            response_language="Español",
            summary="dentista 33012",
            medical_need=True,
            location="33012",
            specialties=["Dentistry"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        with patch.object(agent, "_interpret_user_need", return_value=query):
            result = agent.handle_request(
                _SequencedChatClient(),
                "dentista 33012",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(source.calls), 8)
        self.assertEqual(result.count("provider-card__title"), 2)
        self.assertIn('<div class="provider-card__title">1. Miami Lakes Dentistry Center</div>', result)
        self.assertIn('<div class="provider-card__title">2. Zzz Family Dental</div>', result)


if __name__ == "__main__":
    unittest.main()
