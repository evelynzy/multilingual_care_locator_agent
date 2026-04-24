import sys
import types
import unittest
from typing import Any, Dict, List, Optional, Tuple
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
from provider_search.normalization import build_canonical_provider, build_request_fingerprint
from provider_search.service import ProviderSearchService
from provider_search.sources.clinicaltables import ClinicalTablesSource, DEFAULT_DATASET_CONFIGS


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
            and request.query_filter == "addr_practice.state:NY AND addr_practice.zip:10013*"
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


class _ObgynZipClinicalTablesSource:
    def __init__(
        self,
        noisy_zip_providers: List[Any],
        canonical_term_providers: List[Any],
    ) -> None:
        self.noisy_zip_providers = list(noisy_zip_providers)
        self.canonical_term_providers = list(canonical_term_providers)
        self.calls = []

    def suggest_specialty_terms(self, specialties: Tuple[str, ...]) -> Tuple[str, ...]:
        return tuple(specialty.strip() for specialty in specialties if specialty.strip())

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if dataset == "npi_idv" and (
            request.query_filter == "addr_practice.zip:98101*"
            and request.search_terms == "OB/GYN"
        ):
            return SourceSearchResult(
                providers=list(self.noisy_zip_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.noisy_zip_providers),
                ),
            )
        if request.query_filter == "addr_practice.zip:98101*" and request.search_terms in {
            "Obstetrics & Gynecology",
            "Obstetrics & Gynecology 98101",
        }:
            return SourceSearchResult(
                providers=list(self.canonical_term_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.canonical_term_providers),
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
            request.query_filter == "addr_practice.zip:33012*"
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
            and request.query_filter == "addr_practice.state:TX AND addr_practice.zip:75001*"
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


class _PrimaryCareSynonymClinicalTablesSource:
    def __init__(self, retry_providers: List[Any]) -> None:
        self.retry_providers = list(retry_providers)
        self.calls = []

    def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
        self.calls.append((dataset, request))
        if request.query_filter == "addr_practice.state:TX" and request.search_terms in {
            "Primary Care",
            "Primary Care Dallas",
            "Primary Care TX 75001",
            "Primary Care 75001 Dallas TX",
        }:
            return SourceSearchResult(
                providers=list(self.retry_providers),
                trace=SourceTrace(
                    source_name="clinicaltables",
                    dataset=dataset,
                    status_code=200,
                    result_count=len(self.retry_providers),
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

    def test_handle_request_emits_local_debug_logs_only_under_dual_opt_in(self) -> None:
        noisy_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
            phone="408-555-0102",
        )
        canonical_provider = build_canonical_provider(
            provider_id="1619271780",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="OB/GYN",
            specialties=("OB/GYN",),
            phone="408-555-0100",
        )
        service = ProviderSearchService(
            clinicaltables_source=_ObgynZipClinicalTablesSource(
                noisy_zip_providers=[noisy_provider],
                canonical_term_providers=[canonical_provider],
            ),
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)

        single_gate_client = _ScriptedChatClient(
            [
                {
                    "content": (
                        '{"detected_language":"English","response_language":"English",'
                        '"summary":"ob gyn 98101","medical_need":true,"location":"98101",'
                        '"specialties":["OB/GYN"],"insurance":[],"preferred_languages":[],'
                        '"keywords":[],"patient_context":null}'
                    )
                },
                {"include_choice": False},
            ]
        )

        with patch.object(agent, "_trusted_resource_fallback", return_value=[]):
            with patch.dict("os.environ", {"PROVIDER_SEARCH_DEBUG": "1"}, clear=False):
                with self.assertLogs(level="INFO") as captured_single_gate:
                    agent.handle_request(
                        single_gate_client,
                        "ob gyn 98101",
                        [],
                        max_tokens=256,
                        temperature=0.2,
                        top_p=0.9,
                    )

        single_gate_logs = "\n".join(captured_single_gate.output)
        self.assertNotIn("care_agent_local_debug_interpret", single_gate_logs)
        self.assertNotIn("care_agent_local_debug_handoff", single_gate_logs)
        self.assertNotIn("provider_search_debug_plan", single_gate_logs)
        self.assertNotIn("provider_search_debug_gate_drop", single_gate_logs)
        self.assertNotIn("care_agent_result_debug", single_gate_logs)

        dual_gate_client = _ScriptedChatClient(
            [
                {
                    "content": (
                        '{"detected_language":"English","response_language":"English",'
                        '"summary":"ob gyn 98101","medical_need":true,"location":"98101",'
                        '"specialties":["OB/GYN"],"insurance":[],"preferred_languages":[],'
                        '"keywords":[],"patient_context":null}'
                    )
                },
                {"include_choice": False},
            ]
        )

        with patch.object(agent, "_trusted_resource_fallback", return_value=[]):
            with patch.dict(
                "os.environ",
                {"PROVIDER_SEARCH_DEBUG": "1", "CARE_LOCATOR_LOCAL_DEBUG": "1"},
                clear=False,
            ):
                with self.assertLogs(level="INFO") as captured_dual_gate:
                    agent.handle_request(
                        dual_gate_client,
                        "ob gyn 98101",
                        [],
                        max_tokens=256,
                        temperature=0.2,
                        top_p=0.9,
                    )

        dual_gate_logs = "\n".join(captured_dual_gate.output)
        self.assertIn("care_agent_local_debug_interpret", dual_gate_logs)
        self.assertIn("care_agent_local_debug_handoff", dual_gate_logs)
        self.assertIn("provider_search_debug_plan", dual_gate_logs)
        self.assertIn("provider_search_debug_variant", dual_gate_logs)
        self.assertIn("provider_search_debug_candidate", dual_gate_logs)
        self.assertIn("provider_search_debug_gate_drop", dual_gate_logs)
        self.assertIn("care_agent_result_debug", dual_gate_logs)
        self.assertNotIn("provider_search_debug_candidate_detail", dual_gate_logs)
        self.assertNotIn("Cupertino OB/GYN Associates", dual_gate_logs)
        self.assertNotIn("1619271780", dual_gate_logs)
        self.assertNotIn("408-555-0100", dual_gate_logs)

    def test_handle_request_emits_scoped_candidate_dump_when_fingerprint_matches(self) -> None:
        noisy_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Downtown Imaging Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
            phone="408-555-0102",
        )
        canonical_provider = build_canonical_provider(
            provider_id="1619271780",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="OB/GYN",
            specialties=("OB/GYN",),
            phone="408-555-0100",
        )
        request_fingerprint = build_request_fingerprint(
            ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            )
        )
        service = ProviderSearchService(
            clinicaltables_source=_ObgynZipClinicalTablesSource(
                noisy_zip_providers=[noisy_provider],
                canonical_term_providers=[canonical_provider],
            ),
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=5,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {
                    "content": (
                        '{"detected_language":"English","response_language":"English",'
                        '"summary":"ob gyn 98101","medical_need":true,"location":"98101",'
                        '"specialties":["OB/GYN"],"insurance":[],"preferred_languages":[],'
                        '"keywords":[],"patient_context":null}'
                    )
                },
                {"include_choice": False},
            ]
        )

        with patch.object(agent, "_trusted_resource_fallback", return_value=[]):
            with patch.dict(
                "os.environ",
                {
                    "PROVIDER_SEARCH_DEBUG": "1",
                    "CARE_LOCATOR_LOCAL_DEBUG": "1",
                    "PROVIDER_SEARCH_DEBUG_FINGERPRINT": request_fingerprint,
                },
                clear=False,
            ):
                with self.assertLogs(level="INFO") as captured:
                    agent.handle_request(
                        client,
                        "ob gyn 98101",
                        [],
                        max_tokens=256,
                        temperature=0.2,
                        top_p=0.9,
                    )

        joined_logs = "\n".join(captured.output)
        self.assertIn("provider_search_debug_candidate_detail", joined_logs)
        self.assertIn("gate_outcome=admitted", joined_logs)
        self.assertIn("gate_outcome=dropped", joined_logs)
        self.assertIn("drop_reason=specialty_mismatch", joined_logs)
        self.assertNotIn("Cupertino OB/GYN Associates", joined_logs)
        self.assertNotIn("1619271780", joined_logs)
        self.assertNotIn("408-555-0100", joined_logs)
        self.assertNotIn("98101", joined_logs)

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
            "addr_practice.state:TX AND addr_practice.zip:75001*",
        )
        self.assertEqual(retry_request.query_filter, "addr_practice.state:TX")
        self.assertNotIn("Dallas, TX 75001", [request.search_terms for _, request in source.calls])
        trusted_fallback.assert_not_called()
        self.assertIn("Dallas Family Clinic", result)
        self.assertNotIn("Downtown Imaging Associates", result)
        self.assertNotIn("Medicare Care Compare", result)

    def test_handle_request_primary_care_75001_keeps_live_primary_care_synonym_candidates(self) -> None:
        retry_providers = [
            build_canonical_provider(
                provider_id="provider-clinic-primary-care",
                name="Concentra Primary Care PA",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Dallas",
                state="TX",
                taxonomy="Clinic/Center",
                specialties=("Clinic/Center", "Clinic/Center, Primary Care", "261QP2300X"),
                phone="214-555-0100",
            ),
            build_canonical_provider(
                provider_id="provider-family-med",
                name="North Dallas Primary Care Doctors PLLC",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                city="Dallas",
                state="TX",
                taxonomy="Physician/Internal Medicine",
                specialties=("Physician/Internal Medicine", "Family Medicine", "Internal Medicine"),
                phone="214-555-0101",
            ),
        ]
        source = _PrimaryCareSynonymClinicalTablesSource(retry_providers)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_org",),
            per_dataset_limit=10,
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

        self.assertIn("Concentra Primary Care PA", result)
        self.assertIn("North Dallas Primary Care Doctors PLLC", result)
        self.assertNotIn("Medicare Care Compare", result)
        trusted_fallback.assert_not_called()

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

    def test_handle_request_restores_zip_from_raw_message_when_valid_interpret_json_omits_location(self) -> None:
        obgyn_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Cupertino",
            state="CA",
            taxonomy="OB/GYN",
            specialties=("OB/GYN",),
            phone="408-555-0100",
        )
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            provider_results=(
                ProviderSearchResult(
                    provider=obgyn_provider,
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
                {
                    "content": (
                        '{"detected_language":"English","response_language":"English","summary":"ob gyn 98101",'
                        '"medical_need":true,"location":null,"specialties":["OB/GYN"],"insurance":[],'
                        '"preferred_languages":[],"keywords":[],"patient_context":null,'
                        '"care_setting":"specialist","urgency":null,"needs_clarification":false,'
                        '"follow_up_focus":[]}'
                    ),
                    "finish_reason": "stop",
                }
            ]
        )

        result = agent.handle_request(
            client,
            "ob gyn 98101",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 1)
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("OB/GYN",))
        self.assertEqual(provider_request.location, "98101")
        self.assertIn("Cupertino OB/GYN Associates", result)

    def test_handle_request_keeps_explicit_cpt_intent_when_valid_interpret_json_omits_location(self) -> None:
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(),
            search_trace=SearchTrace(),
        )
        agent = CareLocatorAgent(provider_search_service=service)
        client = _ScriptedChatClient(
            [
                {
                    "content": (
                        '{"detected_language":"English","response_language":"English","summary":"CPT 98101 ob gyn",'
                        '"medical_need":true,"location":null,"specialties":["OB/GYN"],"insurance":[],'
                        '"preferred_languages":[],"keywords":["cpt"],"patient_context":null,'
                        '"care_setting":"specialist","urgency":null,"needs_clarification":false,'
                        '"follow_up_focus":["procedure code"]}'
                    ),
                    "finish_reason": "stop",
                }
            ]
        )

        result = agent.handle_request(
            client,
            "CPT 98101 ob gyn",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.location, None)
        self.assertEqual(provider_request.specialties, ("OB/GYN",))
        self.assertIn("Important safety and trust notes:", result)

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

    def test_handle_request_rescues_obgyn_zip_from_malformed_interpret_json(self) -> None:
        obgyn_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="NPI Registry (individual)",
            dataset="npi_idv",
            city="Cupertino",
            state="CA",
            taxonomy="OB/GYN",
            specialties=("OB/GYN",),
            phone="408-555-0100",
        )
        service = Mock()
        service.search.return_value = ProviderSearchResponse(
            request=ProviderSearchRequest(
                specialties=("OB/GYN",),
                location="98101",
            ),
            provider_results=(
                ProviderSearchResult(
                    provider=obgyn_provider,
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
            "ob gyn 98101",
            [],
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
        )

        self.assertEqual(len(client.calls), 2)
        service.search.assert_called_once()
        provider_request = service.search.call_args[0][0]
        self.assertEqual(provider_request.specialties, ("OB/GYN",))
        self.assertEqual(provider_request.location, "98101")
        self.assertIn("Cupertino OB/GYN Associates", result)

    def test_handle_request_obgyn_98101_continues_to_canonical_variant_after_single_noisy_idv_hit(self) -> None:
        noisy_zip_providers = [
            build_canonical_provider(
                provider_id="provider-noise-0",
                name="Noisy Clinician 0",
                source_name="ClinicalTables",
                dataset="npi_idv",
                city="Santa Clara",
                state="CA",
                specialties=(),
                taxonomy=None,
            )
        ]
        specialty_bearing_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
            phone="408-555-0100",
        )
        source = _ObgynZipClinicalTablesSource(
            noisy_zip_providers,
            [specialty_bearing_provider],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
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
                "ob gyn 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms,
            ["OB/GYN", "Obstetrics & Gynecology"],
        )
        self.assertIn("Cupertino OB/GYN Associates", result)
        self.assertNotIn("Noisy Clinician", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_obgyn_98101_accepts_live_taxonomy_desc_payload_without_fallback(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
                "408-555-0100",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
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
                "ob gyn 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(session.get.call_args_list), 1)
        _, kwargs = session.get.call_args
        self.assertEqual(kwargs["params"]["terms"], "obstetrics gynecology")
        self.assertEqual(kwargs["params"]["q"], "addr_practice.zip:98101*")
        self.assertEqual(
            kwargs["params"]["sf"],
            ",".join(
                [
                    "provider_type",
                    "licenses.medicare.type",
                    "licenses.taxonomy.classification",
                    "licenses.taxonomy.specialization",
                    "licenses.taxonomy.code",
                ]
            ),
        )
        self.assertIn("Cupertino OB/GYN Associates", result)
        self.assertIn("Obstetrics &amp; Gynecology", result)
        self.assertNotIn("Medicare Care Compare", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_obgyn_98101_accepts_code_only_obgyn_payload_without_fallback(self) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
                "408-555-0100",
            ]],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
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
                "ob gyn 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(session.get.call_args_list), 1)
        _, kwargs = session.get.call_args
        self.assertEqual(kwargs["params"]["terms"], "obstetrics gynecology")
        self.assertEqual(kwargs["params"]["q"], "addr_practice.zip:98101*")
        self.assertEqual(
            kwargs["params"]["sf"],
            ",".join(
                [
                    "provider_type",
                    "licenses.medicare.type",
                    "licenses.taxonomy.classification",
                    "licenses.taxonomy.specialization",
                    "licenses.taxonomy.code",
                ]
            ),
        )
        self.assertIn("Cupertino OB/GYN Associates", result)
        self.assertNotIn("Medicare Care Compare", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_obgyn_98101_accepts_descendant_code_candidate_pool_without_fallback(
        self,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            3,
            ["display row 1", "display row 2", "display row 3"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [
                [
                    "Cupertino Gynecology Group",
                    "1619271780",
                    "",
                    "",
                    "207VC0200X",
                    "Santa Clara",
                    "CA",
                    "98101",
                    "408-555-0101",
                ],
                [
                    "South Bay Maternal Fetal Medicine",
                    "1619271781",
                    "",
                    "",
                    "207VM0101X",
                    "Santa Clara",
                    "CA",
                    "98101",
                    "408-555-0102",
                ],
                [
                    "Downtown Imaging Associates",
                    "1619271782",
                    "",
                    "",
                    "2085R0202X",
                    "Santa Clara",
                    "CA",
                    "98101",
                    "408-555-0103",
                ],
            ],
        ]
        response.raise_for_status.return_value = None

        session = Mock()
        session.get.return_value = response
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
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
                "ob gyn 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(session.get.call_args_list), 1)
        _, kwargs = session.get.call_args
        self.assertEqual(kwargs["params"]["terms"], "obstetrics gynecology")
        self.assertEqual(kwargs["params"]["q"], "addr_practice.zip:98101*")
        self.assertEqual(
            kwargs["params"]["sf"],
            ",".join(
                [
                    "provider_type",
                    "licenses.medicare.type",
                    "licenses.taxonomy.classification",
                    "licenses.taxonomy.specialization",
                    "licenses.taxonomy.code",
                ]
            ),
        )
        self.assertIn("Cupertino Gynecology Group", result)
        self.assertIn("South Bay Maternal Fetal Medicine", result)
        self.assertNotIn("Downtown Imaging Associates", result)
        self.assertNotIn("Medicare Care Compare", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_obgyn_98101_uses_demo_broad_recall_variant_after_precise_requests(
        self,
    ) -> None:
        empty_response = Mock()
        empty_response.status_code = 200
        empty_response.json.return_value = [0, [], [], []]
        empty_response.raise_for_status.return_value = None

        success_response = Mock()
        success_response.status_code = 200
        success_response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
                "408-555-0100",
            ]],
        ]
        success_response.raise_for_status.return_value = None

        session = Mock()
        session.get.side_effect = [
            empty_response,
            empty_response,
            empty_response,
            success_response,
        ]
        source = ClinicalTablesSource(session=session)
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
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
                "ob gyn 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(session.get.call_args_list), 4)
        request_params = [kwargs["params"] for _, kwargs in session.get.call_args_list]
        self.assertEqual(request_params[0]["terms"], "obstetrics gynecology")
        self.assertEqual(request_params[0]["q"], "addr_practice.zip:98101*")
        self.assertIn("sf", request_params[0])
        self.assertEqual(request_params[1]["terms"], "obstetrics gynecology")
        self.assertEqual(request_params[1]["q"], "addr_practice.zip:98101*")
        self.assertIn("sf", request_params[1])
        self.assertEqual(request_params[2]["terms"], "obstetrics gynecology")
        self.assertEqual(request_params[2]["q"], "addr_practice.zip:98101*")
        self.assertIn("sf", request_params[2])
        self.assertEqual(request_params[3]["terms"], "ob gyn 98101")
        self.assertNotIn("q", request_params[3])
        self.assertNotIn("sf", request_params[3])
        self.assertIn("Cupertino OB/GYN Associates", result)
        trusted_fallback.assert_not_called()

    def test_default_carelocatoragent_path_uses_shared_clinicaltables_defaults_for_obgyn_98101(
        self,
    ) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = [
            1,
            ["display row"],
            [
                "name.full",
                "NPI",
                "provider_type",
                "taxonomies[0].desc",
                "taxonomies[0].code",
                "addr_practice.city",
                "addr_practice.state",
                "addr_practice.zip",
                "addr_practice.phone",
            ],
            [[
                "Cupertino OB/GYN Associates",
                "1619271780",
                "",
                "Obstetrics & Gynecology",
                "207V00000X",
                "Santa Clara",
                "CA",
                "98101",
                "408-555-0100",
            ]],
        ]
        response.raise_for_status.return_value = None
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
            insurance=[],
            preferred_languages=[],
            keywords=[],
            patient_context=None,
        )

        with patch("provider_search.cache.resolve_provider_cache_path", return_value=None), patch(
            "provider_search.sources.clinicaltables.requests.get",
            return_value=response,
        ) as mocked_get, patch(
            "provider_search.sources.nppes.NPPESSource.enrich_provider",
            autospec=True,
            side_effect=lambda _self, provider: provider,
        ):
            agent = CareLocatorAgent()
            self.assertEqual(
                agent.provider_search_service.clinicaltables_source.dataset_configs,
                DEFAULT_DATASET_CONFIGS,
            )
            self.assertEqual(
                agent.provider_search_service.datasets,
                tuple(DEFAULT_DATASET_CONFIGS.keys()),
            )
            with patch.object(agent, "_interpret_user_need", return_value=query), patch.object(
                agent,
                "_trusted_resource_fallback",
            ) as trusted_fallback:
                result = agent.handle_request(
                    _SequencedChatClient(),
                    "ob gyn 98101",
                    [],
                    max_tokens=256,
                    temperature=0.2,
                    top_p=0.9,
                )

        self.assertGreaterEqual(mocked_get.call_count, 1)
        for _, kwargs in mocked_get.call_args_list:
            self.assertEqual(kwargs["params"]["terms"], "obstetrics gynecology")
            self.assertNotIn("98101", kwargs["params"]["terms"])
            self.assertEqual(kwargs["params"]["q"], "addr_practice.zip:98101*")
            self.assertEqual(
                kwargs["params"]["sf"],
                ",".join(
                    [
                        "provider_type",
                        "licenses.medicare.type",
                        "licenses.taxonomy.classification",
                        "licenses.taxonomy.specialization",
                        "licenses.taxonomy.code",
                    ]
                ),
            )
            self.assertIn("taxonomies[0].desc", kwargs["params"]["df"])
            self.assertIn("taxonomies[0].code", kwargs["params"]["df"])
        self.assertIn("Cupertino OB/GYN Associates", result)
        self.assertIn("Obstetrics &amp; Gynecology", result)
        self.assertNotIn("Medicare Care Compare", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_obgyn_98101_uses_canonical_zip_variant_when_non_location_variants_stay_low_signal(
        self,
    ) -> None:
        noisy_zip_providers = [
            build_canonical_provider(
                provider_id="provider-noise-0",
                name="Noisy Clinician 0",
                source_name="ClinicalTables",
                dataset="npi_idv",
                city="Santa Clara",
                state="CA",
                specialties=(),
                taxonomy=None,
            )
        ]
        specialty_bearing_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
            phone="408-555-0100",
        )

        class _CanonicalZipOnlyObgynSource(_ObgynZipClinicalTablesSource):
            def search_dataset(self, dataset: str, request: Any) -> SourceSearchResult:
                self.calls.append((dataset, request))
                if dataset == "npi_idv" and (
                    request.query_filter == "addr_practice.zip:98101*"
                    and request.search_terms in {"OB/GYN", "Obstetrics & Gynecology"}
                ):
                    return SourceSearchResult(
                        providers=list(self.noisy_zip_providers),
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=len(self.noisy_zip_providers),
                        ),
                    )
                if request.query_filter == "addr_practice.zip:98101*" and (
                    request.search_terms == "Obstetrics & Gynecology 98101"
                ):
                    return SourceSearchResult(
                        providers=list(self.canonical_term_providers),
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=len(self.canonical_term_providers),
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

        source = _CanonicalZipOnlyObgynSource(
            noisy_zip_providers,
            [specialty_bearing_provider],
        )
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
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
                "ob gyn 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms,
            ["OB/GYN", "Obstetrics & Gynecology", "Obstetrics & Gynecology 98101"],
        )
        self.assertIn("Cupertino OB/GYN Associates", result)
        self.assertNotIn("Noisy Clinician", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_obgyn_98101_does_not_stop_on_unrelated_specialty_bearing_first_hit(
        self,
    ) -> None:
        unrelated_provider = build_canonical_provider(
            provider_id="provider-radiology",
            name="Santa Clara Imaging Group",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Diagnostic Radiology",
            specialties=("Diagnostic Radiology",),
        )
        specialty_bearing_provider = build_canonical_provider(
            provider_id="provider-obgyn",
            name="Cupertino OB/GYN Associates",
            source_name="ClinicalTables",
            dataset="npi_idv",
            city="Santa Clara",
            state="CA",
            taxonomy="Obstetrics & Gynecology",
            specialties=("Obstetrics & Gynecology",),
            phone="408-555-0100",
        )

        class _UnrelatedSpecialtyFirstHitObgynSource(_ObgynZipClinicalTablesSource):
            def search_dataset(self, dataset: Any, request: Any) -> SourceSearchResult:
                self.calls.append((dataset, request))
                if dataset == "npi_idv" and (
                    request.query_filter == "addr_practice.zip:98101*"
                    and request.search_terms == "OB/GYN"
                ):
                    return SourceSearchResult(
                        providers=[unrelated_provider],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=1,
                        ),
                    )
                if request.query_filter == "addr_practice.zip:98101*" and (
                    request.search_terms == "Obstetrics & Gynecology"
                ):
                    return SourceSearchResult(
                        providers=[specialty_bearing_provider],
                        trace=SourceTrace(
                            source_name="clinicaltables",
                            dataset=dataset,
                            status_code=200,
                            result_count=1,
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

        source = _UnrelatedSpecialtyFirstHitObgynSource([], [])
        service = ProviderSearchService(
            clinicaltables_source=source,
            cache=None,
            datasets=("npi_idv",),
            per_dataset_limit=20,
        )
        agent = CareLocatorAgent(provider_search_service=service)
        query = ParsedCareQuery(
            detected_language="English",
            response_language="English",
            summary="ob gyn 98101",
            medical_need=True,
            location="98101",
            specialties=["OB/GYN"],
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
                "ob gyn 98101",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        searched_terms = [request.search_terms for _, request in source.calls]
        self.assertEqual(
            searched_terms,
            ["OB/GYN", "Obstetrics & Gynecology"],
        )
        self.assertIn("Cupertino OB/GYN Associates", result)
        self.assertNotIn("Santa Clara Imaging Group", result)
        trusted_fallback.assert_not_called()

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
        self.assertEqual(len(source.calls), 1)
        _, first_request = source.calls[0]
        self.assertEqual(first_request.search_terms, "Dentistry")
        self.assertEqual(first_request.query_filter, "addr_practice.zip:33012*")
        self.assertIn("Florida Children&#x27;s Dentistry, P.A.", result)
        self.assertNotIn("Miami Lakes Dentistry Center", result)

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
            "addr_practice.state:NY AND addr_practice.zip:10013*",
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

    def test_handle_request_dentista_33012_admits_local_dentistry_descendants_before_fallback(self) -> None:
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

        self.assertEqual(len(source.calls), 1)
        self.assertIn("Florida Children&#x27;s Dentistry, P.A.", result)
        self.assertIn("Hialeah Square Dentistry, PA", result)
        self.assertIn("Caplin and Gober Dentistry, PA", result)
        self.assertNotIn("Miami Lakes Dentistry Center", result)
        self.assertNotIn("Medicare Care Compare", result)
        trusted_fallback.assert_not_called()

    def test_handle_request_dentista_33012_renders_two_cards_when_local_duplicate_pair_crowds_limit(self) -> None:
        local_zip_providers = [
            build_canonical_provider(
                provider_id="provider-local-individual",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (individual)",
                dataset="npi_idv",
                address="123 Palm Ave",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Pediatric Dentistry"),
                phone="305-555-0101",
            ),
            build_canonical_provider(
                provider_id="provider-local-org",
                name="Florida Children's Dentistry, P.A.",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                address="123 Palm Ave",
                city="Hialeah",
                state="FL",
                taxonomy="Dentist",
                specialties=("Dentist", "Dentist, Pediatric Dentistry"),
                phone="305-555-0101",
            ),
            build_canonical_provider(
                provider_id="provider-local-second",
                name="Zzz Family Dental",
                source_name="NPI Registry (organization)",
                dataset="npi_org",
                address="900 Pine St",
                city="Miami",
                state="FL",
                taxonomy="Dentistry",
                specialties=("Dentistry",),
                phone="305-555-0102",
            ),
        ]
        source = _NearbyDentalClinicalTablesSource(
            local_zip_providers,
            [],
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

        self.assertEqual(len(source.calls), 2)
        self.assertEqual(result.count("provider-card__title"), 2)
        self.assertIn('<div class="provider-card__title">1. Florida Children&#x27;s Dentistry, P.A.</div>', result)
        self.assertIn('<div class="provider-card__title">2. Zzz Family Dental</div>', result)


if __name__ == "__main__":
    unittest.main()
