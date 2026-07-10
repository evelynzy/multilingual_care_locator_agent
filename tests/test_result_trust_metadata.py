import sys
import types
import unittest
from typing import Optional
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

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")

    class _StubDataset(list):
        pass

    def _stub_load_dataset(*args, **kwargs):
        raise RuntimeError("datasets.load_dataset stubbed in tests")

    datasets_stub.Dataset = _StubDataset
    datasets_stub.load_dataset = _stub_load_dataset
    sys.modules["datasets"] = datasets_stub

if "llama_index" not in sys.modules:
    llama_index_stub = types.ModuleType("llama_index")
    llama_core_stub = types.ModuleType("llama_index.core")

    class _StubDocument:
        def __init__(self, text: str, metadata: Optional[dict] = None):
            self.text = text
            self.metadata = metadata or {}

    class _StubVectorStoreIndex:
        def __init__(self, documents=None, embed_model=None):
            self._documents = documents or []
            self._embed_model = embed_model

        @classmethod
        def from_documents(cls, documents, embed_model=None):
            return cls(documents=documents, embed_model=embed_model)

        def as_retriever(self, similarity_top_k: int = 5):
            class _StubRetriever:
                def retrieve(self_inner, query):
                    return []

            return _StubRetriever()

    llama_core_stub.Document = _StubDocument
    llama_core_stub.VectorStoreIndex = _StubVectorStoreIndex

    llama_embeddings_stub = types.ModuleType("llama_index.embeddings")
    llama_hf_stub = types.ModuleType("llama_index.embeddings.huggingface")

    class _StubHuggingFaceEmbedding:
        def __init__(self, model_name: str):
            self.model_name = model_name

    llama_hf_stub.HuggingFaceEmbedding = _StubHuggingFaceEmbedding

    sys.modules["llama_index"] = llama_index_stub
    sys.modules["llama_index.core"] = llama_core_stub
    sys.modules["llama_index.embeddings"] = llama_embeddings_stub
    sys.modules["llama_index.embeddings.huggingface"] = llama_hf_stub


from care import CareLocatorAgent, ParsedCareQuery
from care.rendering import _DETERMINISTIC_RENDER_COPY, _DETERMINISTIC_RENDER_TRANSLATIONS
from care.safety import _REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE
from provider_search.models import (
    ProviderSearchRequest,
    ProviderSearchResponse,
    ProviderSearchResult,
    SearchTrace,
)
from provider_search.normalization import build_canonical_provider
from retriever import ProviderRecord, ProviderRepository


class _SequencedChatClient:
    def __init__(self, response_texts):
        if isinstance(response_texts, str):
            response_texts = [response_texts]
        self.response_texts = list(response_texts)
        self.calls = []

    def chat_completion(self, messages, max_tokens, temperature, top_p, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        call_number = len(self.calls)
        response_index = min(call_number - 1, len(self.response_texts) - 1)
        response_text = self.response_texts[response_index]
        return type(
            "Completion",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": {"content": response_text},
                            "finish_reason": "stop",
                        },
                    )()
                ]
            },
        )()


class CareLocatorAgentResultTrustMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = CareLocatorAgent(provider_search_service=Mock())

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
        self.agent.provider_search_service.search.return_value = ProviderSearchResponse(
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
            search_trace=SearchTrace(total_candidates=1),
        )

    @staticmethod
    def _client_with_response(response_text: str):
        return _SequencedChatClient(response_text)

    def test_handle_request_normalizes_provider_search_service_trust_metadata(self) -> None:
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

        client = _SequencedChatClient("Model table should not be used.")

        with patch.object(
            self.agent,
            "_interpret_user_need",
            return_value=query,
        ):
            result = self.agent.handle_request(
                client,
                "primary care 94110",
                [],
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )

        self.assertEqual(len(client.calls), 0)
        self.assertIn('<div class="provider-card">', result)
        self.assertIn('<div class="provider-card__title">1. Harmony Family Clinic</div>', result)
        self.assertIn('<div class="provider-card__subtitle">Primary Care • San Francisco, CA</div>', result)
        self.assertIn('Phone</span><span class="provider-card__meta-value">415-555-0100</span>', result)
        self.assertIn('Source</span><span class="provider-card__meta-value">NPI Registry (individual)</span>', result)
        self.assertIn('<span class="provider-card__badge">Informational</span>', result)
        self.assertIn('<span class="provider-card__badge">Network unverified</span>', result)
        self.assertIn('<span class="provider-card__badge">New patients unknown</span>', result)
        self.assertIn('<span class="provider-card__badge">Appointments unverified</span>', result)
        self.assertNotIn('<span class="provider-card__badge">Source: NPI Registry (individual)</span>', result)
        self.assertIn('Why matched</span><span class="provider-card__value">Relevant to your search for primary care in San Francisco.</span>', result)
        self.assertIn('Listed insurance</span><span class="provider-card__value">Medicare, Aetna (reported only; network participation is not verified here)</span>', result)
        self.assertNotIn('Insurance/network verification</span>', result)
        self.assertNotIn('Accepting new patients</span>', result)
        self.assertIn('Appointment availability</span><span class="provider-card__value">Not verified; call the provider to confirm.</span>', result)
        self.assertIn('Next step</span><span class="provider-card__value">Call to confirm network status, referral needs, new-patient status, and appointment availability.</span>', result)
        self.assertIn("Important safety and trust notes:", result)
        self.assertNotIn("### 1. Harmony Family Clinic", result)
        self.agent.provider_search_service.search.assert_called_once()

    def test_provider_record_uses_reported_insurance_metadata(self) -> None:
        record = ProviderRecord(
            id="provider_001",
            name="Harmony Family Clinic",
            specialties=["Primary Care"],
            languages=["English", "Spanish"],
            accepted_insurance=["Medicare", "Aetna"],
            address="123 Valencia St",
            city="San Francisco",
            state="CA",
            country="USA",
            phone="+1-415-555-0100",
            website="https://example.com",
            telehealth=True,
            description="Community-focused care",
        )

        data = record.to_dict()
        document = record.as_document()

        self.assertEqual(data["insurance_reported"], ["Medicare", "Aetna"])
        self.assertEqual(data["insurance_network_verification"]["status"], "unverified")
        self.assertFalse(data["insurance_network_verification"]["verified"])
        self.assertEqual(data["accepting_new_patients_status"]["status"], "unknown")
        self.assertFalse(data["accepting_new_patients_status"]["verified"])
        self.assertEqual(data["provenance"]["source"], "Local provider dataset")
        self.assertIn("Listed insurance (reported, not verified)", document.text)
        self.assertEqual(document.metadata["insurance_reported"], ["Medicare", "Aetna"])

    def test_provider_repository_accepts_reported_insurance_source_field(self) -> None:
        repository = ProviderRepository.__new__(ProviderRepository)

        record = repository._normalize_row(
            {
                "id": "provider_002",
                "name": "Reported Insurance Clinic",
                "specialties": ["Primary Care"],
                "languages": ["English"],
                "insurance_reported": ["Medicaid"],
                "address": "10 Main St",
                "city": "Austin",
                "state": "TX",
                "country": "USA",
            }
        )

        self.assertEqual(record.accepted_insurance, ["Medicaid"])
        self.assertEqual(record.to_dict()["insurance_reported"], ["Medicaid"])

    def test_trust_labels_include_medicare_opt_out_status(self) -> None:
        result = self.agent._normalize_result_trust_metadata(
            {
                "name": "Specialty Clinic",
                "source": "NPI Registry",
                "insurance_reported": ["Medicare"],
                "insurance_network_verification": {"status": "unverified"},
                "accepting_new_patients_status": {"status": "unknown"},
                "medicare_opt_out": {"opted_out": True},
            }
        )

        self.assertIn("Source: NPI Registry", result["trust_labels"])
        self.assertIn("Insurance/network: unverified", result["trust_labels"])
        self.assertIn("New patients: unknown", result["trust_labels"])
        self.assertIn("Medicare opt-out: opted out", result["trust_labels"])

    def test_styled_provider_card_preserves_dynamic_trust_labels(self) -> None:
        result = self.agent._normalize_result_trust_metadata(
            {
                "name": "Specialty Clinic",
                "location": "Austin, TX",
                "phone": "512-555-0100",
                "taxonomy": "Cardiology",
                "source": "NPI Registry",
                "insurance_reported": ["Medicare"],
                "insurance_network_verification": {"status": "unverified"},
                "accepting_new_patients_status": {"status": "unknown"},
                "medicare_opt_out": {"opted_out": True},
            }
        )

        card_html = self.agent._format_provider_result_card(
            result,
            index=1,
            query={"specialties": ["Cardiology"], "keywords": []},
        )

        self.assertIn('<span class="provider-card__badge">Informational</span>', card_html)
        self.assertIn('<span class="provider-card__badge">Network unverified</span>', card_html)
        self.assertIn('<span class="provider-card__badge">New patients unknown</span>', card_html)
        self.assertIn('<span class="provider-card__badge">Appointments unverified</span>', card_html)
        self.assertNotIn('<span class="provider-card__badge">Source: NPI Registry</span>', card_html)
        self.assertNotIn('<span class="provider-card__badge">Insurance/network: unverified</span>', card_html)
        self.assertNotIn('<span class="provider-card__badge">New patients: unknown</span>', card_html)
        self.assertIn('<span class="provider-card__badge">Medicare opt-out: opted out</span>', card_html)
        self.assertIn('Why matched</span><span class="provider-card__value">Listed under Cardiology.</span>', card_html)
        self.assertIn('Listed insurance</span><span class="provider-card__value">Medicare (reported only; network participation is not verified here)</span>', card_html)
        self.assertNotIn('Insurance/network verification</span>', card_html)
        self.assertNotIn('Accepting new patients</span>', card_html)
        self.assertIn('Appointment availability</span><span class="provider-card__value">Not verified; call the provider to confirm.</span>', card_html)

    def test_provider_card_omits_listed_insurance_when_no_insurance_is_reported(self) -> None:
        card_html = self.agent._format_provider_result_card(
            self.agent._normalize_result_trust_metadata(
                {
                    "name": "Fallback Clinic",
                    "specialties": ["Primary Care"],
                    "location": "Austin, TX",
                    "phone": "512-555-0199",
                    "source": "Local provider dataset",
                }
            ),
            index=1,
            query={"summary": "primary care"},
        )

        self.assertNotIn("Listed insurance</span>", card_html)

    def test_provider_card_omits_low_signal_subtitle_fragments(self) -> None:
        card_html = self.agent._format_provider_result_card(
            self.agent._normalize_result_trust_metadata(
                {
                    "name": "Fallback Clinic",
                    "specialties": ["M"],
                    "location": "BOYD",
                    "phone": "512-555-0199",
                    "source": "Local provider dataset",
                }
            ),
            index=1,
            query={"summary": "child care"},
        )

        self.assertNotIn('provider-card__subtitle">M • BOYD</div>', card_html)
        self.assertNotIn('provider-card__subtitle">', card_html)

    def test_compose_result_card_response_localizes_chinese_deterministic_copy(self) -> None:
        payload = {
            "query": {
                "response_language": "中文",
                "summary": "儿科10013",
            },
            "care_setting_guidance": "For routine or ongoing care, primary care is usually the best fit.",
            "specialist_plan_guidance": "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents.",
            "local_results": [
                self.agent._normalize_result_trust_metadata(
                    {
                        "name": "Harmony Family Clinic",
                        "specialties": ["Pediatrics"],
                        "location": "New York, NY",
                        "phone": "212-555-0100",
                        "source": "Local provider dataset",
                        "insurance_reported": ["Medicaid"],
                    }
                )
            ],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.",
        }

        response = self.agent._compose_result_card_response(payload)

        zh = _DETERMINISTIC_RENDER_COPY["simplified_chinese"]
        t = lambda s: _DETERMINISTIC_RENDER_TRANSLATIONS[s]["simplified_chinese"]
        self.assertIn(zh["results_intro"].format(summary="儿科10013"), response)
        self.assertIn(
            "**{0}:** {1}".format(
                zh["care_route_label"],
                t("For routine or ongoing care, primary care is usually the best fit."),
            ),
            response,
        )
        self.assertIn(
            "**{0}:** {1}".format(
                zh["referral_note_label"],
                t("For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents."),
            ),
            response,
        )
        self.assertIn(zh["phone_label"] + "</span>", response)
        self.assertIn(zh["why_matched_label"] + "</span>", response)
        self.assertIn(zh["next_step_label"] + "</span>", response)
        self.assertIn(zh["matched_search_summary"].format(summary="儿科10013"), response)
        self.assertIn(zh["verification_reminder_short"], response)
        self.assertIn(
            '{0}</span><span class="provider-card__value">Medicaid ({1})</span>'.format(
                zh["listed_insurance_label"], zh["listed_insurance_suffix"]
            ),
            response,
        )
        self.assertNotIn(zh["insurance_verification_label"] + "</span>", response)
        self.assertNotIn(zh["accepting_patients_label"] + "</span>", response)
        self.assertIn(
            '{0}</span><span class="provider-card__value">{1}</span>'.format(
                zh["appointment_availability_label"], zh["appointment_availability_value"]
            ),
            response,
        )
        self.assertIn(
            "**{0}:** {1}".format(zh["before_contact_label"], zh["verification_reminder"]),
            response,
        )
        self.assertIn(
            '{0}</span><span class="provider-card__meta-value">Local provider dataset</span>'.format(
                zh["source_label"]
            ),
            response,
        )
        self.assertNotIn("Here are care navigation results for", response)
        self.assertNotIn("**Care route:**", response)
        self.assertNotIn("**Referral note:**", response)
        self.assertNotIn("Why matched", response)
        self.assertNotIn("Pediatrics, pediatric, child health", response)


    def test_compose_result_card_response_localizes_spanish_deterministic_copy(self) -> None:
        payload = {
            "query": {
                "response_language": "Español",
                "summary": "atención primaria en Austin",
            },
            "care_setting_guidance": "For a known specialty or referral need, a specialist is usually the right route.",
            "specialist_plan_guidance": "For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents.",
            "local_results": [
                self.agent._normalize_result_trust_metadata(
                    {
                        "name": "Specialty Clinic",
                        "taxonomy": "Cardiology",
                        "location": "Austin, TX",
                        "phone": "512-555-0100",
                        "source": "NPI Registry",
                        "insurance_reported": ["Medicare"],
                        "medicare_opt_out": {"opted_out": True},
                    }
                )
            ],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status, accepted insurance plan, referral requirements, new-patient availability, location, and appointment availability.",
        }

        response = self.agent._compose_result_card_response(payload)

        es = _DETERMINISTIC_RENDER_COPY["spanish"]
        t = lambda s: _DETERMINISTIC_RENDER_TRANSLATIONS[s]["spanish"]
        self.assertIn(es["results_intro"].format(summary="atención primaria en Austin"), response)
        self.assertIn(
            "**{0}:** {1}".format(
                es["care_route_label"],
                t("For a known specialty or referral need, a specialist is usually the right route."),
            ),
            response,
        )
        self.assertIn(
            "**{0}:** {1}".format(
                es["referral_note_label"],
                t("For specialist searches, HMO and POS plans often require a PCP referral; PPO plans may not, but you should confirm the rule with your insurer and plan documents."),
            ),
            response,
        )
        self.assertIn(es["phone_label"] + "</span>", response)
        self.assertIn(es["why_matched_label"] + "</span>", response)
        self.assertIn(es["next_step_label"] + "</span>", response)
        self.assertIn(es["matched_search_summary"].format(summary="atención primaria en Austin"), response)
        self.assertIn(es["verification_reminder_short"], response)
        self.assertIn(
            '{0}</span><span class="provider-card__value">Medicare ({1})</span>'.format(
                es["listed_insurance_label"], es["listed_insurance_suffix"]
            ),
            response,
        )
        self.assertNotIn(es["insurance_verification_label"] + "</span>", response)
        self.assertNotIn(es["accepting_patients_label"] + "</span>", response)
        self.assertIn(
            '{0}</span><span class="provider-card__value">{1}</span>'.format(
                es["appointment_availability_label"], es["appointment_availability_value"]
            ),
            response,
        )
        self.assertIn(
            "**{0}:** {1}".format(es["before_contact_label"], es["verification_reminder"]),
            response,
        )
        self.assertIn(
            es["trust_label_medicare_opt_out"].format(value=es["medicare_opted_out"]),
            response,
        )
        self.assertNotIn("Here are care navigation results for", response)
        self.assertNotIn("**Care route:**", response)
        self.assertNotIn("**Referral note:**", response)
        self.assertNotIn("Why matched", response)


    def test_compose_result_card_response_uses_localized_match_reason_instead_of_raw_keywords(self) -> None:
        payload = {
            "query": {
                "response_language": "中文",
                "summary": "儿童保健",
                "specialties": ["Pediatrics"],
                "keywords": ["pediatric", "child health"],
            },
            "local_results": [
                self.agent._normalize_result_trust_metadata(
                    {
                        "name": "Harmony Family Clinic",
                        "specialties": ["Pediatrics"],
                        "location": "New York, NY",
                        "phone": "212-555-0100",
                        "source": "Local provider dataset",
                    }
                )
            ],
            "fallback_results": [],
        }

        response = self.agent._compose_result_card_response(payload)

        zh = _DETERMINISTIC_RENDER_COPY["simplified_chinese"]
        self.assertIn(
            '{0}</span><span class="provider-card__value">{1}</span>'.format(
                zh["why_matched_label"],
                zh["matched_search_summary"].format(summary="儿童保健"),
            ),
            response,
        )
        self.assertNotIn("Pediatrics, pediatric, child health", response)

    def test_compose_result_card_response_renders_fallback_resources_in_separate_section(self) -> None:
        payload = {
            "query": {
                "response_language": "English",
                "summary": "primary care in Denver",
            },
            "local_results": [
                self.agent._normalize_result_trust_metadata(
                    {
                        "name": "Riverside Family Medicine",
                        "specialties": ["Primary Care"],
                        "location": "Denver, CO",
                        "phone": "412-555-0100",
                        "source": "Local provider dataset",
                    }
                )
            ],
            "fallback_results": [
                {
                    "name": "Medicare Care Compare",
                    "location": "Denver, CO",
                    "website": "https://www.medicare.gov/care-compare/",
                    "description": "Compare clinicians and facilities using Medicare's public directory.",
                    "source": "Trusted public directories",
                }
            ],
        }

        response = self.agent._compose_result_card_response(payload)

        self.assertIn('provider-card__title">1. Riverside Family Medicine</div>', response)
        self.assertIn("**Trusted resources and fallback options:**", response)
        self.assertIn("1. **Medicare Care Compare**: Region: Denver, CO", response)
        self.assertIn("; Source: Trusted public directories", response)
        self.assertIn(r"; Website: https://www.medicare.gov/care-compare/", response)
        self.assertIn(
            r"; Details: Compare clinicians and facilities using Medicare's public directory\.",
            response,
        )
        self.assertNotIn('provider-card__title">2. Medicare Care Compare</div>', response)

    def test_compose_result_card_response_does_not_apply_provider_defaults_to_fallback_only_resources(self) -> None:
        payload = {
            "query": {
                "response_language": "English",
                "summary": "primary care in Denver",
            },
            "local_results": [],
            "fallback_results": [
                {
                    "name": "Medicare Care Compare",
                    "location": "Denver, CO",
                    "website": "https://www.medicare.gov/care-compare/",
                    "description": "Compare clinicians and facilities using Medicare's public directory.",
                    "source": "Trusted public directories",
                }
            ],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_result_card_response(payload)

        self.assertIn("**Trusted resources and fallback options:**", response)
        self.assertIn("Medicare Care Compare", response)
        self.assertNotIn("provider-card__", response)
        self.assertNotIn("Why matched</span>", response)
        self.assertNotIn("Appointment availability</span>", response)
        self.assertNotIn("Listed insurance</span>", response)
        self.assertNotIn("**Before you contact a provider:**", response)
        self.assertNotIn("Call the provider and insurer to confirm network status.", response)

    def test_compose_result_card_response_escapes_fallback_resource_fields(self) -> None:
        payload = {
            "query": {
                "response_language": "English",
                "summary": "primary care in Denver",
            },
            "local_results": [],
            "fallback_results": [
                {
                    "name": "Medicare **Care** Compare [Official]",
                    "location": "Denver, CO (Metro)",
                    "website": "https://example.com/care(compare)",
                    "description": "Use *trusted* public data > call first.",
                    "source": "Trusted [public] directories",
                }
            ],
        }

        response = self.agent._compose_result_card_response(payload)

        self.assertIn(r"1. **Medicare \*\*Care\*\* Compare \[Official\]**", response)
        self.assertIn(r"Region: Denver, CO \(Metro\)", response)
        self.assertIn(r"Source: Trusted \[public\] directories", response)
        self.assertIn("Website: https://example.com/care(compare)", response)
        self.assertIn(r"Details: Use \*trusted\* public data \> call first\.", response)
        self.assertNotIn("Medicare **Care** Compare [Official]", response)
        self.assertNotIn("Use *trusted* public data > call first.", response)
        self.assertNotIn(r"Website: https://example\.com/care\(compare\)", response)

    def test_format_fallback_resource_entry_escapes_non_url_website_text(self) -> None:
        entry = self.agent._format_fallback_resource_entry(
            {
                "name": "Fallback Directory",
                "location": "Austin, TX",
                "website": "Portal [login]",
                "description": "Bring plan details.",
                "source": "Trusted public directories",
            },
            1,
        )

        self.assertIn(r"Website: Portal \[login\]", entry)
        self.assertNotIn("Website: Portal [login]", entry)

    def test_format_fallback_resource_entry_escapes_non_web_url_schemes(self) -> None:
        entry = self.agent._format_fallback_resource_entry(
            {
                "name": "Fallback Directory",
                "location": "Austin, TX",
                "website": "javascript://alert(1)",
                "description": "Bring plan details.",
                "source": "Trusted public directories",
            },
            1,
        )

        self.assertIn(r"Website: javascript://alert\(1\)", entry)
        self.assertNotIn("Website: javascript://alert(1)", entry)

        self.assertEqual(
            self.agent._format_visible_website_value("https://example.com/path(test)"),
            "https://example.com/path(test)",
        )
        self.assertEqual(
            self.agent._format_visible_website_value("http://example.com/path(test)"),
            "http://example.com/path(test)",
        )
        self.assertEqual(
            self.agent._format_visible_website_value("data://payload[test](here)"),
            r"data://payload\[test\]\(here\)",
        )
        self.assertEqual(
            self.agent._format_visible_website_value("file://tmp/report(backup)"),
            r"file://tmp/report\(backup\)",
        )

    def test_compose_response_appends_required_trust_guidance(self) -> None:
        client = self._client_with_response("Model-rendered answer.")
        payload = {
            "query": {
                "response_language": "English",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertIn("Model-rendered answer.", response)
        self.assertIn("Important safety and trust notes:", response)
        self.assertIn("Directory matches are informational", response)
        self.assertIn("Call the provider and insurer to confirm", response)
        self.assertIn("Do not share personal health information", response)
        self.assertEqual(len(client.calls), 1)

    def test_compose_response_uses_prewritten_required_trust_guidance_for_supported_languages(self) -> None:
        supported_languages = [
            ("English", "english"),
            ("Español", "spanish"),
            ("中文", "simplified_chinese"),
            ("Vietnamese", "vietnamese"),
            ("Tagalog", "tagalog"),
            ("Filipino", "tagalog"),
            ("Arabic", "arabic"),
            ("Korean", "korean"),
        ]

        for language, language_key in supported_languages:
            with self.subTest(language=language):
                client = self._client_with_response("Model-rendered answer.")
                payload = {
                    "query": {
                        "response_language": language,
                        "medical_need": True,
                        "summary": "primary care",
                    },
                    "local_results": [],
                    "fallback_results": [],
                    "verification_guidance": "Call the provider and insurer to confirm network status.",
                }

                response = self.agent._compose_response(
                    client,
                    payload,
                    max_tokens=128,
                    temperature=0.1,
                    top_p=0.9,
                )

                self.assertIn("Model-rendered answer.", response)
                self.assertIn(_REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE[language_key], response)
                # the footer must appear exactly once (dedup guard)
                self.assertEqual(
                    response.count(_REQUIRED_TRUST_GUIDANCE_BY_LANGUAGE[language_key]), 1
                )


    def test_compose_response_uses_same_deterministic_supported_language_note_each_time(self) -> None:
        client = self._client_with_response("Respuesta generada por el modelo.")
        payload = {
            "query": {
                "response_language": "Español",
                "medical_need": True,
                "summary": "atencion primaria",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        first_response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )
        second_response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertEqual(first_response, second_response)
        self.assertIn("Notas importantes de seguridad y confianza:", first_response)
        self.assertEqual(len(client.calls), 2)

    def test_compose_response_falls_back_to_english_for_unsupported_detected_language_without_translation(self) -> None:
        client = _SequencedChatClient(["Réponse du modèle.", "Unexpected translated note"])
        payload = {
            "query": {
                "detected_language": "French",
                "medical_need": True,
                "summary": "soins primaires",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertIn("Réponse du modèle.", response)
        self.assertIn("Important safety and trust notes:", response)
        self.assertIn("Directory matches are informational", response)
        self.assertEqual(response.count("Important safety and trust notes:"), 1)
        self.assertNotIn("Unexpected translated note", response)
        self.assertEqual(len(client.calls), 1)

    def test_compose_response_falls_back_to_english_trust_guidance_for_unknown_language(self) -> None:
        client = self._client_with_response("Model answer.")
        payload = {
            "query": {
                "response_language": "unknown",
                "medical_need": True,
                "summary": "primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        response = self.agent._compose_response(
            client,
            payload,
            max_tokens=128,
            temperature=0.1,
            top_p=0.9,
        )

        self.assertIn("Important safety and trust notes:", response)
        self.assertIn("Directory matches are informational", response)
        self.assertEqual(response.count("Important safety and trust notes:"), 1)
        self.assertEqual(len(client.calls), 1)

    def test_compose_response_logs_omit_phi_bearing_payload(self) -> None:
        client = type(
            "Client",
            (),
            {
                "chat_completion": lambda self, messages, max_tokens, temperature, top_p: type(
                    "Completion",
                    (),
                    {
                        "choices": [
                            type(
                                "Choice",
                                (),
                                {
                                    "message": {"content": "Model answer."},
                                    "finish_reason": "stop",
                                },
                            )()
                        ]
                    },
                )()
            },
        )()
        payload = {
            "query": {
                "response_language": "English",
                "medical_need": True,
                "summary": "Jane Doe at 123 Main St needs primary care",
            },
            "local_results": [],
            "fallback_results": [],
            "verification_guidance": "Call the provider and insurer to confirm network status.",
        }

        with self.assertLogs("care.agent", level="DEBUG") as captured_logs:
            self.agent._compose_response(
                client,
                payload,
                max_tokens=128,
                temperature=0.1,
                top_p=0.9,
            )

        log_output = "\n".join(captured_logs.output)
        self.assertNotIn("Jane Doe", log_output)
        self.assertNotIn("123 Main St", log_output)


if __name__ == "__main__":
    unittest.main()
